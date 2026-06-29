# LT3.c

LT3 attention-loop-only GDN 语言模型：基于 ClimbMix 训练和 CORE 评测脚本，token mixer 使用 `flash-linear-attention` 的 `GatedDeltaNet`。每层内共享同一套 GDN 参数重复执行 `loop_count` 次 attention，FFN 每层只执行一次。

所有命令默认从本子目录执行：

```bash
cd LT3.c
```

创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 训练

### 1) 准备数据

数据统一下载到 `data/`。默认数据集为 **ClimbMix**（parquet 分片）：

```bash
python climbmix.py download -n 170
```

也可使用 TinyStories：

```bash
python tinystories.py download
```

> **切换数据集**：修改 `train.py` 顶部的 `from climbmix import Task`（例如改成 `from tinystories import Task`）。两个数据集的 `Task` 接口一致。

### 2) 预分词

Llama 2 词表（vocab_size=32000）：

```bash
python climbmix.py pretokenize
```

自训练词表（vocab_size=512，主要用于 TinyStories 小模型）：

```bash
python tinystories.py pretokenize --vocab_size=512
```

### 3) 训练

训练输出写入 `out/`（或 `--out_dir` 指定目录），自动保存 `ckpt.pt` 和 `model.bin`。

#### 15M attention-loop GDN-only（~15M 物理参数，loop_count=4）

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --out_dir=outputs/climbmix15M \
  --dim=288 --n_layers=6 --num_heads=4 \
  --head_k_dim=72 --head_v_dim=72 --hidden_dim=768 --conv_size=4 \
  --loop_count=4 --use_residual=True \
  --max_seq_len=1024 \
  --batch_size=64 --gradient_accumulation_steps=8 \
  --learning_rate=5e-4 --max_iters=100000 --warmup_iters=1000 \
  --weight_decay=0.1 --beta1=0.9 --beta2=0.95 --grad_clip=1.0 \
  --dtype=bfloat16 --compile=False \
  --eval_interval=2000 --eval_iters=100
```

#### 42M
```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --out_dir=outputs/climbmix42M \
  --dim=512 --n_layers=8 --num_heads=8 \
  --head_k_dim=64 --head_v_dim=64 --hidden_dim=1376 --conv_size=4 \
  --loop_count=4 --use_residual=True \
  --max_seq_len=1024 \
  --batch_size=32 --gradient_accumulation_steps=16 \
  --learning_rate=5e-4 --max_iters=100000 --warmup_iters=1000 \
  --weight_decay=0.1 --beta1=0.9 --beta2=0.95 --grad_clip=1.0 \
  --dtype=bfloat16 --compile=False \
  --eval_interval=2000 --eval_iters=100
```
#### 110M

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --out_dir=outputs/climbmix110M \
  --dim=768 --n_layers=12 --num_heads=12 \
  --head_k_dim=64 --head_v_dim=64 --hidden_dim=2048 --conv_size=4 \
  --loop_count=4 --use_residual=True \
  --max_seq_len=1024 \
  --batch_size=16 --gradient_accumulation_steps=32 \
  --learning_rate=5e-4 --max_iters=100000 --warmup_iters=1000 \
  --weight_decay=0.1 --beta1=0.9 --beta2=0.95 --grad_clip=1.0 \
  --dtype=bfloat16 --compile=False \
  --eval_interval=2000 --eval_iters=100
```

#### 340M

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --out_dir=outputs/climbmix340M \
  --dim=1024 --n_layers=21 --num_heads=6 \
  --head_k_dim=256 --head_v_dim=512 --hidden_dim=4096 --conv_size=4 \
  --loop_count=4 --use_residual=True \
  --norm_eps=1e-6 \
  --max_seq_len=4096 \
  --batch_size=2 --gradient_accumulation_steps=64 \
  --learning_rate=3e-4 --max_iters=200000 --warmup_iters=10000 \
  --weight_decay=0.1 --beta1=0.9 --beta2=0.95 --grad_clip=1.0 \
  --dtype=bfloat16 --compile=True \
  --eval_interval=5000 --eval_iters=50
```

#### 1B

```bash
torchrun --standalone --nproc_per_node=8 train.py \
  --out_dir=outputs/climbmix1B \
  --dim=2048 --n_layers=21 --num_heads=6 \
  --head_k_dim=256 --head_v_dim=512 --hidden_dim=8192 --conv_size=4 \
  --loop_count=4 --use_residual=True \
  --norm_eps=1e-6 \
  --max_seq_len=4096 \
  --batch_size=4 --gradient_accumulation_steps=32 \
  --learning_rate=3e-4 --max_iters=200000 --warmup_iters=10000 \
  --weight_decay=0.1 --beta1=0.9 --beta2=0.95 --grad_clip=1.0 \
  --dtype=bfloat16 --compile=True \
  --eval_interval=5000 --eval_iters=50
```

---

## C 推理

当前 `run.c` 仍是 template 的 Llama runner，不能正确读取 attention-loop GDN 权重。训练和 CORE 评测使用 PyTorch checkpoint。`export.py` 会保存一个归档式 `.bin`，供后续实现 attention-loop GDN C runner 时对齐权重布局。

## CORE 评测

```bash
python core_eval.py \
  --checkpoint outputs/climbmix15M_50B_100k/ckpt.pt \
  --eval_batch_size=8 \
  --head_stat_forwards=512
```

结果写入 checkpoint 同目录：

- `core_eval.csv`
- `ckpt.bin`
- `gdn_head_stats.json`
- `gdn_head_stats.csv`
- `gdn_head_stats_report.md`

`gdn_head_stats_report.md` 会按 `(attention_loop, layer, head)` 输出 GDN 的 head 统计：

- `beta`: `sigmoid(b_proj)`，近似 update strength
- `g_silu_abs`: `SiLU(g_proj)` 按 head 分组后的平均绝对值
- `a_raw`: `a_proj` 原始均值，和 decay/forget dynamics 相关

这些不是 softmax attention entropy；线性注意力没有显式 attention weight matrix。

## 架构说明

| 组件 | 实现 |
|------|------|
| Mixer | `fla.layers.GatedDeltaNet` |
| Loop | 每层内同一套 GDN attention 固定重复 `loop_count` 次 |
| Loss | 最终 hidden state 上单次 CE |
| Loop residual | 可选 per-dimension attention-loop residual gate，零初始化 |
| FFN | `fla.modules.GatedMLP`，每层只运行一次 |
| Norm | RMSNorm |
| 词表 | weight tying |

## 参考论文以及仓库

```bash
mkdir references
cd references
git clone git@github.com:chili-lab/LT2.git #证明循环线性注意力+FFN有用
git clone git@github.com:YaoChen0203/Sparse-Growing-Transformer.git #证明只循环注意力有用
git clone git@github.com:ruhai-lin/gdn.c.git #线性注意力
git clone git@github.com:ruhai-lin/ouro.c.git #循环模型
```

## 性能对比

ClimbMix 上训练 100k iter（~50B tokens）后的 [CORE](https://arxiv.org/abs/2406.11717) benchmark 结果。对比四种架构：

| 模型 | 循环方式 | 物理参数 |
|------|----------|----------|
| [gdn.c](../references/gdn.c/) | 无循环（GDN baseline） | 15M / 42M / 110M |
| [LT2.c](../LT2.c/) | 全层 loop×4 | 15M / 42M / 110M |
| LT3.c | 仅 attention loop×4，FFN 不循环 | 15M / 42M / 110M |
| [Ouro-1.4B](../ouro.c/) | 预训练循环 Transformer（参考） | 1.4B |

### CORE 汇总（Centered）

| 规模 | gdn.c | LT2.c | LT3.c | LT3 − gdn.c | LT3 − LT2 |
|------|-------|-------|-------|-------------|-----------|
| 15M | 0.0501 | 0.0552 | **0.0652** | +0.0151 | +0.0100 |
| 42M | 0.0918 | 0.1072 | **0.1122** | +0.0204 | +0.0050 |
| 110M | 0.1416 | 0.1752 | 0.1556 | +0.0140 | −0.0197 |
| 1.4B (Ouro) | — | — | — | — | CORE **0.5458** |

15M / 42M 上，LT3 的 attention-only loop 均优于 gdn.c baseline 和 LT2 全层 loop；110M 上 LT3 仍优于 gdn.c，但低于 LT2 全层 loop。

### 各子任务 Centered 分数

| Task | gdn 15M | LT2 15M | LT3 15M | gdn 42M | LT2 42M | LT3 42M | gdn 110M | LT2 110M | LT3 110M | Ouro 1.4B |
|------|---------|---------|---------|---------|---------|---------|----------|----------|----------|-----------|
| hellaswag_zeroshot | 0.0420 | 0.0461 | 0.0407 | 0.0973 | 0.1209 | 0.1038 | 0.1962 | 0.2448 | 0.2113 | 0.6192 |
| jeopardy | 0.0014 | 0.0005 | 0.0019 | 0.0043 | 0.0076 | 0.0019 | 0.0132 | 0.0279 | 0.0241 | 0.4454 |
| bigbench_qa_wikidata | 0.0842 | 0.0638 | 0.0156 | 0.2008 | 0.1800 | 0.2434 | 0.3148 | 0.3815 | 0.3530 | 0.6888 |
| arc_easy | 0.1801 | 0.1908 | 0.1667 | 0.2710 | 0.3075 | 0.2710 | 0.3911 | 0.4506 | 0.4186 | 0.7525 |
| arc_challenge | −0.0307 | −0.0262 | −0.0398 | −0.0068 | −0.0023 | −0.0102 | 0.0341 | 0.0774 | 0.0694 | 0.4369 |
| copa | −0.1600 | −0.0800 | −0.1600 | −0.0200 | −0.1000 | −0.0600 | 0.0200 | 0.1200 | 0.0800 | 0.6200 |
| commonsense_qa | 0.0121 | −0.0012 | 0.0213 | 0.1462 | 0.0152 | 0.1554 | 0.0070 | 0.1421 | 0.1042 | 0.7584 |
| piqa | 0.2013 | 0.2078 | 0.1948 | 0.2546 | 0.3025 | 0.2775 | 0.3765 | 0.3830 | 0.3776 | 0.5996 |
| openbook_qa | 0.0373 | 0.0480 | 0.0587 | 0.0587 | 0.0827 | 0.0960 | 0.1067 | 0.1467 | 0.1173 | 0.2373 |
| lambada_openai | 0.1304 | 0.1467 | 0.1502 | 0.2141 | 0.2342 | 0.2313 | 0.3060 | 0.3235 | 0.3066 | 0.6507 |
| hellaswag | 0.0339 | 0.0419 | 0.0383 | 0.0918 | 0.1151 | 0.0996 | 0.1889 | 0.2429 | 0.2064 | 0.6426 |
| winograd | 0.0037 | 0.0696 | 0.0989 | 0.1282 | 0.1941 | 0.1868 | 0.1209 | 0.1868 | 0.1648 | 0.6337 |
| winogrande | −0.0182 | −0.0355 | 0.0481 | −0.0024 | 0.0560 | 0.0592 | 0.0418 | 0.0450 | 0.0276 | 0.3307 |
| bigbench_dyck_languages | 0.0110 | 0.0070 | 0.0890 | 0.0320 | 0.0670 | 0.0580 | 0.1500 | 0.1990 | 0.0980 | 0.2750 |
| agi_eval_lsat_ar | 0.0707 | 0.0435 | 0.0543 | 0.0272 | 0.0435 | 0.0489 | 0.0870 | 0.0707 | 0.0380 | 0.1739 |
| bigbench_cs_algorithms | 0.3947 | 0.3848 | 0.3705 | 0.4197 | 0.3614 | 0.3780 | 0.4508 | 0.4318 | 0.4303 | 0.7227 |
| bigbench_operators | 0.1381 | 0.0905 | 0.1095 | 0.1286 | 0.1238 | 0.1048 | 0.1476 | 0.1524 | 0.1238 | 0.8333 |
| bigbench_repeat_copy_logic | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0313 | 0.0000 | 0.4063 |
| squad | 0.0099 | 0.0168 | 0.0132 | 0.0346 | 0.0592 | 0.0472 | 0.0645 | 0.0979 | 0.0803 | 0.6494 |
| coqa | 0.0441 | 0.0556 | 0.0551 | 0.0864 | 0.0997 | 0.0968 | 0.1156 | 0.1419 | 0.1260 | 0.4773 |
| boolq | −0.2643 | −0.2401 | −0.0687 | −0.3222 | −0.0888 | −0.1057 | −0.1975 | −0.2160 | −0.1210 | 0.6274 |
| bigbench_language_identification | 0.1796 | 0.1838 | 0.1756 | 0.1765 | 0.1803 | 0.1839 | 0.1801 | 0.1740 | 0.1859 | 0.4264 |
| **CORE** | **0.0501** | **0.0552** | **0.0652** | **0.0918** | **0.1072** | **0.1122** | **0.1416** | **0.1752** | **0.1556** | **0.5458** |