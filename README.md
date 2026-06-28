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
  --out_dir=outputs/climbmix15M_50B_100k \
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

当前先只固定 15M 规模，确认只循环 GDN attention、FFN 不循环时在这个 token budget 下是否是正收益。42M/110M 建议等 15M 的 CORE 和 head stats 跑通后再按同样比例扩展 `dim/n_layers/hidden_dim/num_heads`。

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