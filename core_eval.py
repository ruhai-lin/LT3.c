"""
CORE metric evaluation for LT3.c attention-loop GDN checkpoints.

Example:
    python core_eval.py --checkpoint outputs/climbmix15M_50B_100k/ckpt.pt --eval_batch_size 8

Outputs next to the checkpoint:
  - core_eval.csv
  - <checkpoint_stem>.bin
  - gdn_head_stats.{json,csv,md}
"""

import argparse
import csv
import json
import os
import random
import shutil
import tempfile
import time
import zipfile
from collections import defaultdict

import requests
import torch
import torch.nn.functional as F
from jinja2 import Template
from tqdm import tqdm
import yaml

from export import load_checkpoint, model_export
from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {'fewshot_examples': fewshot_examples, 'continuation_delimiter': continuation_delimiter, 'item': item}
    return [template.render(choice=choice, **context) for choice in item['choices']]


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {'fewshot_examples': fewshot_examples, 'continuation_delimiter': continuation_delimiter, 'item': item}
    return [template.render(context=context_option, **context) for context_option in item['context_options']]


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {'fewshot_examples': fewshot_examples, 'continuation_delimiter': continuation_delimiter, 'item': item}
    prompt_without = template.render(include_continuation=False, **context).strip()
    prompt_with = template.render(include_continuation=True, **context)
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    min_len = min(len(seq) for seq in token_sequences)
    indices = {'left': range(min_len), 'right': range(-1, -min_len - 1, -1)}[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    answer_start_idx = find_common_length(tokens, direction='left')
    return tokens, [answer_start_idx] * len(prompts), [len(x) for x in tokens]


def batch_sequences_schema(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx = find_common_length([tokens_without, tokens_with], direction='left')
    return [tokens_with], [start_idx], [len(tokens_with)]


class GDNHeadObserver:
    """Records loop/layer/head statistics from GDN projections during eval.

    This is not attention entropy. It tracks linear-attention head behavior:
    - beta: sigmoid(b_proj), update strength per head
    - a_raw: raw a_proj values, related to decay/forget dynamics
    - g_silu_abs: mean absolute SiLU(g_proj) grouped by value head
    """

    def __init__(self, model, enabled=True, max_forwards=512):
        self.model = model
        self.enabled = enabled
        self.max_forwards = max_forwards
        self.forward_count = 0
        self.module_calls = defaultdict(int)
        self.sums = defaultdict(float)
        self.sq_sums = defaultdict(float)
        self.counts = defaultdict(int)
        self.handles = []
        self.failures = defaultdict(int)

    def should_collect(self):
        return self.enabled and (self.max_forwards < 0 or self.forward_count < self.max_forwards)

    def mark_forward(self):
        if self.should_collect():
            self.forward_count += 1

    def attach(self):
        if not self.enabled:
            return
        for name, module in self.model.named_modules():
            if name.endswith(".mixer.b_proj") or name.endswith(".mixer.a_proj") or name.endswith(".mixer.g_proj"):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))
        print(f"Attached GDN observer hooks to {len(self.handles)} projection modules")

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _make_hook(self, name):
        def hook(module, inputs, output):
            if not self.should_collect():
                return
            try:
                parts = name.split(".")
                layer_idx = int(parts[1])
                proj = parts[-1]
                loop_idx = self.module_calls[name] % max(self.model.params.loop_count, 1)
                self.module_calls[name] += 1
                x = output.detach().float()
                if proj == "b_proj":
                    values = torch.sigmoid(x).mean(dim=(0, 1))
                    metric = "beta"
                elif proj == "a_proj":
                    values = x.mean(dim=(0, 1))
                    metric = "a_raw"
                elif proj == "g_proj":
                    bsz, seqlen, dim = x.shape
                    h = self.model.params.num_heads
                    values = F.silu(x).abs().view(bsz, seqlen, h, -1).mean(dim=(0, 1, 3))
                    metric = "g_silu_abs"
                else:
                    return
                for head, val in enumerate(values.cpu().tolist()):
                    key = (metric, loop_idx, layer_idx, head)
                    self.sums[key] += float(val)
                    self.sq_sums[key] += float(val) * float(val)
                    self.counts[key] += 1
            except Exception as exc:
                self.failures[f"{type(exc).__name__}: {str(exc)[:120]}"] += 1
        return hook

    def rows(self):
        out = []
        for key in sorted(self.counts):
            metric, loop_idx, layer_idx, head = key
            n = self.counts[key]
            mean = self.sums[key] / n
            var = max(self.sq_sums[key] / n - mean * mean, 0.0)
            out.append({
                "metric": metric,
                "loop": loop_idx,
                "layer": layer_idx,
                "head": head,
                "mean": mean,
                "std": var ** 0.5,
                "count": n,
            })
        return out


def write_gdn_report(out_dir, observer):
    rows = observer.rows()
    json_path = os.path.join(out_dir, "gdn_head_stats.json")
    csv_path = os.path.join(out_dir, "gdn_head_stats.csv")
    md_path = os.path.join(out_dir, "gdn_head_stats_report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "failures": dict(observer.failures)}, f, indent=2)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "loop", "layer", "head", "mean", "std", "count"])
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# GDN Head Stats Report\n\n")
        f.write("These are loop/layer/head statistics for GDN projections during CORE eval. They are not softmax attention entropy.\n\n")
        f.write(f"Observed forwards: {observer.forward_count}\n\n")
        for metric in ["beta", "g_silu_abs", "a_raw"]:
            metric_rows = [r for r in rows if r["metric"] == metric]
            f.write(f"## Top {metric}\n\n")
            f.write("| Rank | Loop | Layer | Head | Mean | Std |\n|---:|---:|---:|---:|---:|---:|\n")
            reverse = metric != "a_raw"
            for rank, r in enumerate(sorted(metric_rows, key=lambda x: x["mean"], reverse=reverse)[:30], start=1):
                f.write(f"| {rank} | {r['loop']} | {r['layer']} | {r['head']} | {r['mean']:.6f} | {r['std']:.6f} |\n")
            f.write("\n")
        if observer.failures:
            f.write("## Hook Failures\n\n")
            for msg, count in observer.failures.items():
                f.write(f"- {count}x `{msg}`\n")

    print(f"GDN head stats written to: {json_path}")
    print(f"GDN head stats CSV written to: {csv_path}")
    print(f"Readable GDN report written to: {md_path}")


@torch.no_grad()
def forward_model(model, input_ids):
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    losses = F.cross_entropy(outputs.view(batch_size * seq_len, -1).float(), target_ids.view(batch_size * seq_len), reduction='none').view(batch_size, seq_len)
    losses[:, -1] = float('nan')
    predictions = outputs.argmax(dim=-1)
    observer = getattr(model, "gdn_observer", None)
    if observer is not None:
        observer.mark_forward()
    return losses, predictions


@torch.no_grad()
def evaluate_task_batched(model, tokenizer, data, device, task_meta, eval_batch_size):
    pad_token_id = tokenizer.get_bos_token_id()
    records, sequences, start_idxs, end_idxs = [], [], [], []

    for idx, item in enumerate(data):
        task_type = task_meta['task_type']
        num_fewshot = task_meta['num_fewshot']
        continuation_delimiter = task_meta['continuation_delimiter']
        fewshot_examples = []
        if num_fewshot > 0:
            rng = random.Random(1234 + idx)
            available_indices = [i for i in range(len(data)) if i != idx]
            fewshot_indices = rng.sample(available_indices, min(num_fewshot, len(available_indices)))
            fewshot_examples = [data[i] for i in fewshot_indices]

        if task_type == 'multiple_choice':
            toks, starts, ends = batch_sequences_mc(tokenizer, render_prompts_mc(item, continuation_delimiter, fewshot_examples))
        elif task_type == 'schema':
            toks, starts, ends = batch_sequences_schema(tokenizer, render_prompts_schema(item, continuation_delimiter, fewshot_examples))
        elif task_type == 'language_modeling':
            toks, starts, ends = batch_sequences_lm(tokenizer, render_prompts_lm(item, continuation_delimiter, fewshot_examples))
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        max_tokens = getattr(model, 'max_seq_len', None)
        if max_tokens is not None:
            new_toks, new_starts, new_ends = [], [], []
            for t, s, e in zip(toks, starts, ends):
                if len(t) > max_tokens:
                    crop = len(t) - max_tokens
                    new_toks.append(t[-max_tokens:])
                    new_start = max(s - crop, 1)
                    new_end = max(e - crop, new_start)
                    new_starts.append(new_start)
                    new_ends.append(new_end)
                else:
                    new_toks.append(t)
                    new_starts.append(s)
                    new_ends.append(e)
            toks, starts, ends = new_toks, new_starts, new_ends

        seq_indices = []
        for t, s, e in zip(toks, starts, ends):
            seq_indices.append(len(sequences))
            sequences.append(t)
            start_idxs.append(s)
            end_idxs.append(e)
        records.append({"task_type": task_type, "gold": item.get("gold"), "seq_indices": seq_indices})

    seq_losses = [None] * len(sequences)
    seq_predictions = [None] * len(sequences)
    for start in range(0, len(sequences), eval_batch_size):
        end = min(start + eval_batch_size, len(sequences))
        input_ids = stack_sequences(sequences[start:end], pad_token_id).to(device)
        losses, predictions = forward_model(model, input_ids)
        for local_idx, seq_idx in enumerate(range(start, end)):
            si, ei = start_idxs[seq_idx], end_idxs[seq_idx]
            seq_losses[seq_idx] = losses[local_idx, si - 1:ei - 1].detach().cpu()
            seq_predictions[seq_idx] = predictions[local_idx].detach().cpu()

    correct = 0.0
    for rec in records:
        if rec["task_type"] == 'language_modeling':
            seq_idx = rec["seq_indices"][0]
            si, ei = start_idxs[seq_idx], end_idxs[seq_idx]
            pred = seq_predictions[seq_idx][si - 1:ei - 1]
            actual = torch.tensor(sequences[seq_idx][si:ei], dtype=pred.dtype)
            correct += float(torch.all(pred == actual).item())
        else:
            mean_losses = []
            for seq_idx in rec["seq_indices"]:
                loss_slice = seq_losses[seq_idx]
                mean_losses.append(loss_slice.mean().item() if loss_slice.numel() else float("inf"))
            correct += float(mean_losses.index(min(mean_losses)) == rec["gold"])
    return correct / len(data)


class ModelWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.max_seq_len = model.params.max_seq_len
        self.gdn_observer = None

    def __call__(self, input_ids):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            return self.model(input_ids, targets=input_ids)


class TokenizerWrapper:
    def __init__(self, enc):
        self.enc = enc

    def __call__(self, prompts, prepend=None):
        bos = prepend is not None
        return [self.enc.encode(p, bos=bos, eos=False) for p in prompts]

    def get_bos_token_id(self):
        return self.enc.bos_id


def place_eval_bundle(zip_path, eval_bundle_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        shutil.move(os.path.join(tmpdir, "eval_bundle"), eval_bundle_dir)


def ensure_eval_bundle():
    eval_bundle_dir = os.path.join(DATA_CACHE_DIR, "eval_bundle")
    if os.path.exists(eval_bundle_dir):
        return eval_bundle_dir
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_CACHE_DIR, "eval_bundle.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading eval bundle from {EVAL_BUNDLE_URL}...")
        resp = requests.get(EVAL_BUNDLE_URL, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(total=total, unit="iB", unit_scale=True) as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                bar.update(f.write(chunk))
    print("Unpacking eval bundle...")
    place_eval_bundle(zip_path, eval_bundle_dir)
    return eval_bundle_dir


def evaluate_core(model, tokenizer, device, max_per_task=-1, eval_batch_size=8):
    eval_bundle_dir = ensure_eval_bundle()
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        tasks = yaml.safe_load(f)['icl_tasks']
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            random_baselines[row['Eval Task']] = float(row['Random baseline'])

    results, centered_results = {}, {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {'task_type': task['icl_task_type'], 'num_fewshot': task['num_fewshot'][0], 'continuation_delimiter': task.get('continuation_delimiter', ' ')}
        data_path = os.path.join(data_base_path, task['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]
        random.Random(1337).shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]
        accuracy = evaluate_task_batched(model, tokenizer, data, device, task_meta, eval_batch_size)
        results[label] = accuracy
        rb = random_baselines[label]
        centered = (accuracy - 0.01 * rb) / (1.0 - 0.01 * rb)
        centered_results[label] = centered
        elapsed = time.time() - start_time
        print(f"  {label:<35} acc {accuracy:.4f} | centered {centered:.4f} | {task_meta['num_fewshot']}-shot {task_meta['task_type']} | {elapsed:.1f}s")
    return {"results": results, "centered_results": centered_results, "core_metric": sum(centered_results.values()) / len(centered_results)}


def main():
    parser = argparse.ArgumentParser(description="CORE metric evaluation for LT3 attention-loop GDN")
    parser.add_argument("--checkpoint", default=os.path.join("outputs", "climbmix15M_50B_100k", "ckpt.pt"))
    parser.add_argument("--tokenizer", default="tokenizer.model")
    parser.add_argument("--max_per_task", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no_head_stats", action="store_true")
    parser.add_argument("--head_stat_forwards", type=int, default=512)
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.checkpoint)) or "."
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    model = load_checkpoint(args.checkpoint).to(args.device)
    model.eval()
    wrapped_model = ModelWrapper(model, args.device)
    observer = GDNHeadObserver(model, enabled=not args.no_head_stats, max_forwards=args.head_stat_forwards)
    observer.attach()
    wrapped_model.gdn_observer = observer
    model.gdn_observer = observer
    enc = Tokenizer(args.tokenizer)
    wrapped_tokenizer = TokenizerWrapper(enc)

    bin_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(args.checkpoint))[0]}.bin")
    model_export(model, bin_path, version=0)

    try:
        core = evaluate_core(wrapped_model, wrapped_tokenizer, args.device, max_per_task=args.max_per_task, eval_batch_size=args.eval_batch_size)
    finally:
        observer.close()

    csv_path = os.path.join(out_dir, "core_eval.csv")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Accuracy", "Centered"])
        for label in core["results"]:
            writer.writerow([label, f"{core['results'][label]:.6f}", f"{core['centered_results'][label]:.6f}"])
        writer.writerow(["CORE", "", f"{core['core_metric']:.6f}"])
    print(f"\nCORE metric: {core['core_metric']:.4f}")
    print(f"Results written to: {csv_path}")
    print(f"Checkpoint exported to: {bin_path}")
    if not args.no_head_stats:
        write_gdn_report(out_dir, observer)


if __name__ == "__main__":
    main()
