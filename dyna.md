# Experiment Plan: Head Differentiation Across Scales

## Background & Motivation

### RLKV 的核心发现

RLKV (arxiv:2510.08525, own work) 用 RL (GRPO) 作为探针，发现 reasoning LLM 中只有少数 attention heads 需要完整 KV cache 来维持推理一致性（称为 reasoning-critical heads），其余可以激进压缩为 sink + local window，甚至压缩后 pass@1 性能反而变好。

关键观察：
1. **Head 功能分化是 task-independent 的**——学到的是一个跨 task 通用的 static head mask
2. **分化在 base model 就存在**——rebuttal 实验在 Qwen-2.5-Math-7B 和 Qwen-3-4B-Base 上验证，RLKV 识别的 heads 在 base model 阶段就比随机 heads 更重要（压缩后性能下降更剧烈）
3. **压缩 non-critical heads 后性能反而变好**——说明 full attention 对这些 heads 引入了噪声，sink + local window 反而帮它们"聚焦"
4. **Reasoning heads ≠ retrieval heads**——DuoAttention 用 retrieval 能力识别的 heads 和 RLKV 用 reasoning 质量识别的 heads 不同

### Hybrid-Head Architecture 的尝试与困境

基于 RLKV 的发现，我们尝试在 pre-training 阶段直接将 head 分为 full attention 和 sliding window attention 两类（hybrid-head architecture）。小规模实验（d16, 8 heads, 1024 dim, 2k seq）结果：

| Metric | Full baseline | Hybrid (4 full + 4 SW-128) | Delta |
|--------|---------------|---------------------------|-------|
| CORE | 0.1941 | 0.2021 | +4.1% |
| BPB | 0.792 | 0.793 | -0.1% |

下游 eval 有提升，但**模型没有学到预期的 head 分化 pattern**。大模型（8B）自然涌现的 head 功能分层，在小模型上从头训练时无法复现。

### 核心问题

**Head 功能分化是否有 scale threshold？** 大模型有足够冗余让不同 heads 自然分化为不同角色，小模型是否没有这个能力？

### 为什么用 Fast KVzip 做这个实验

Fast KVzip (arxiv:2601.17668) 提供了一个比 RLKV 更轻量的 head 分化探针：

| | RLKV | Fast KVzip |
|--|------|------------|
| 训练方式 | RL (GRPO) | Forward pass + BCE loss |
| 训练成本 | 22-40 A100 GPU-hours (7-8B) | <1 H100 hour (8-14B) |
| 粒度 | head-level, static | token × head, input-dependent |
| 输出 | 每个 head 一个 α ∈ [0,1] | 每个 token 每个 head 一个 importance score |

Fast KVzip 的 gate 架构中有一个 input-independent 的成分——per-head bias $b_j$：

$$s = \frac{\exp(q^T k)}{\exp(q^T k) + \sum_r \exp(q^T k_\text{sink}^r) + b_j}$$

$b_j$ 大的 head 不管输入什么都会被压缩，等价于 RLKV 的 static pattern。训完后直接看 $b_j$ 分布就能得到 input-independent 的 head 分化信息。

同时，跨多种输入计算 per-head retention rate 的方差，可以验证 input-dependent gate 的输出是否在 head level 上跨输入稳定。

## Experiment Plan

### Step 1: Fast KVzip Gate Training on Qwen3 Series

**模型**：Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B（base model，非 thinking/instruct 版本）

**训练设置**（参考 Fast KVzip 论文）：
- 数据：FineWeb-Edu，1M tokens，序列长度 10K-30K + concatenated 100K
- Gate 架构：Low-rank sink attention，S=16 sink keys，D'=16
- 训练：BCE loss，lr=0.2，5K steps，batch size 1K
- LLM 权重完全冻结，只训 gate 参数
- 预计每个模型 <1 H100 hour（小模型可能更快）

**Target score**：用 KVzip 的 reconstruction-based importance score（Forward pass 计算，不需要 backprop through LLM）

**实现参考**：
- Fast KVzip 没有开源代码（截至 2026-04），需要自行实现
- 核心组件：(1) KVzip reconstruction 计算 target score, (2) sink-attention gate 架构, (3) 逐层独立训练
- 或者用 RLKV 的框架也行，只是训练成本更高

### Step 2: Analyze Head Differentiation Patterns

对每个模型，提取以下信息：

**2a. Input-independent 分化（$b_j$ 分布）**
- 提取每层每个 KV head 的 learned bias $b_j$
- 画 heatmap：x 轴 = layer，y 轴 = head，颜色 = $b_j$ 值
- 对比不同 scale 的 $b_j$ 分布：是否小模型 $b_j$ 集中在中间（无分化），大模型呈现 bimodal（有分化）

**2b. Aggregated retention rate**
- 在多种输入（FineWeb, code, math, QA）上跑 gate
- 计算每个 head 的平均 retention rate 和跨输入方差
- 按 Fast KVzip Section 5 的分类：sparse (r<0.05), medium (0.05≤r<0.9), dense (r≥0.9)
- 对比不同 scale 的 sparse/medium/dense head 比例

**2c. 可视化**
- 每个模型一张 head retention rate heatmap（类似 Fast KVzip Figure 7）
- 三个模型的对比图：head retention rate 的直方图叠在一起
- $b_j$ 分布的直方图对比

### Step 3: 判断是否继续

**如果小模型有分化**（0.6B 或 1.7B 就能看到 bimodal 的 retention 分布）：
- 用发现的 pattern 指导 hybrid architecture：将 sparse heads 指定为 SW，dense heads 指定为 full attention
- 在相同规模上从头训练 hybrid model，对比 baseline
- 看这种 informed assignment 是否优于随机 assignment 和 uniform full attention

**如果小模型没有分化**（只在 4B 才出现）：
- 记录 scale threshold
- 考虑放弃 hybrid-head 方向在小模型上的探索
- 这个 negative result 本身有参考价值，但可能不足以支撑一篇论文

## Important Notes

### 关于 Fast KVzip 实现
- https://arxiv.org/abs/2601.17668
- https://github.com/Janghyun1230/FastKVzip

### 关于模型选择

- 使用 base model，不是 instruct/thinking 版本——目的是看 pre-training 是否产生分化
- Qwen3 系列的好处是架构一致（GQA），不同 scale 之间只有 layer 数、hidden dim、head 数的差异，可控性好
- 如果资源允许，加上 Qwen3-8B 作为 reference point（和 RLKV 在 reasoning model 上的发现对照）

### Qwen3 模型架构参数（已验证，来源：HuggingFace config.json）

| Model | Layers | Hidden Dim | Attn Heads | KV Heads | Head Dim | GQA Ratio | FFN Dim |
|-------|--------|-----------|------------|----------|----------|-----------|---------|
| Qwen3-0.6B | 28 | 1024 | 16 | 8 | 128 | 2:1 | 3072 |
| Qwen3-1.7B | 28 | 2048 | 16 | 8 | 128 | 2:1 | 6144 |
| Qwen3-4B | 36 | 2560 | 32 | 8 | 128 | 4:1 | 9728 |
| Qwen3-8B | 36 | 4096 | 32 | 8 | 128 | 4:1 | 12288 |

注意：
- 所有模型 KV heads 都是 8，head_dim 都是 128。head-level 分化的"空间"在所有 scale 上相同，排除了"head 数量不够所以没有分化"的混淆因素
- 0.6B 的 hidden_size (1024) ≠ num_heads × head_dim (16×128=2048)，Q/K/V 投影维度独立于 hidden_size
- 0.6B/1.7B/4B 使用 tied word embeddings，8B 不使用
- 0.6B → 1.7B 主要 scale width（hidden dim 翻倍），layers 不变；1.7B → 4B 同时 scale depth（28→36）和 width；4B → 8B 只 scale width

### 这个实验的定位

这是一个**低成本的探索性实验**，目的是：
1. 验证 head 分化是否有 scale threshold
2. 积累对不同 scale 下 head 行为的第一手数据
3. 决定 hybrid-head 方向是否值得继续投入

不要带着"一定要发论文"的心态做。先看数据，结果 surprising 再考虑下一步。

---

## Experiment Results (2026-04-09)

### 实验执行

**环境**：1×H100 80G（feature collection + gate training + retention）, 240G CPU RAM 节点（4B gate training）

**Step 1: Gate Training** — 完成 ✅

在 FineWeb-Edu 数据上（29 × 10K-30K tokens + 5 × 100K concatenated）收集 hidden states + KVzip importance scores，然后训练 gate。

| Model | Layers | KV Heads | GQA | Gate File | 训练时间 |
|-------|--------|----------|-----|-----------|---------|
| Qwen3-0.6B | 28 | 8 | 2:1 | `q2_dim16_sink16.pt` (22M) | ~30 min |
| Qwen3-1.7B | 28 | 8 | 2:1 | `q2_dim16_sink16.pt` (22M) | ~40 min |
| Qwen3-4B | 36 | 8 | 4:1 | `q4_dim16_sink16.pt` (45M) | ~50 min |

注意：4B 的 hidden states 加载需要 ~177GB CPU RAM（36 layers × 960K tokens × 2560 dim × bf16），80G 节点 OOM，需要 240G 节点。去掉 `fineweb_10k_cat` 可降至 ~80GB。

**Step 2: Retention Rate Analysis** — 完成 ✅

按 Fast KVzip 论文 Section 5 / Figure 14 的方法，在 SCBench 数据上（`scbench_kv` 50 条 English book + `scbench_repoqa` 50 条 code）计算 per-head retention rate，全局 36% KV budget。

### Step 2a: Per-head bias b_j

| Model | b_j mean | b_j std | b_j range |
|-------|----------|---------|-----------|
| Qwen3-0.6B | -0.0895 | 0.1398 | [-0.78, 0.40] |
| Qwen3-1.7B | -0.0722 | 0.1402 | [-0.64, 0.43] |
| Qwen3-4B | -0.0130 | 0.0927 | [-0.35, 0.36] |

观察：0.6B/1.7B 的 b_j 范围更大（更极端的 input-independent 偏好），4B 反而更集中。

### Step 2b: Retention Rate (SCBench, 36% budget)

**Overall（book + code 合并）**：

| Model | Sparse (<0.05) | Medium (0.05-0.9) | Dense (≥0.9) |
|-------|----------------|-------------------|---------------|
| Qwen3-0.6B | 25.4% | 57.6% | 17.0% |
| Qwen3-1.7B | 28.1% | 54.0% | 17.9% |
| Qwen3-4B | 23.6% | 57.6% | 18.8% |

**Per-domain breakdown**：

| Model | Domain | Sparse | Dense | Cross-example std |
|-------|--------|--------|-------|-------------------|
| Qwen3-0.6B | book (kv) | 41.5% | 19.6% | 0.0021 |
| Qwen3-0.6B | code (repoqa) | 20.5% | 20.1% | 0.0317 |
| Qwen3-1.7B | book (kv) | 42.0% | 20.5% | 0.0024 |
| Qwen3-1.7B | code (repoqa) | 22.8% | 20.1% | 0.0271 |
| Qwen3-4B | book (kv) | 43.8% | 19.1% | 0.0025 |
| Qwen3-4B | code (repoqa) | 17.7% | 19.8% | 0.0345 |

### 核心发现

**1. Head 分化没有 scale threshold**

三个 scale（0.6B, 1.7B, 4B）的 sparse/medium/dense 比例惊人地一致（~25%/55%/18%）。0.6B 就已经有清晰的 bimodal retention 分布。这否定了"小模型没有足够冗余产生 head 分化"的假设。

**2. 分化是 task-independent 的**

Cross-example std 极低（book: ~0.002, code: ~0.03），说明同一个 head 在不同输入上的 retention 行为高度一致。这和 RLKV 的发现一致——head 功能分化是 static 的，不随输入变化。

**3. Book vs Code 的差异**

Book 数据的 sparse head 比例（~42%）显著高于 code（~20%）。Code 数据激活了更多 head（sparse 比例下降），可能因为代码的结构信息（缩进、括号、变量名）需要更多 head 参与 attention。Dense head 比例在两个域上稳定（~20%），说明 dense heads 的重要性不受数据类型影响。

**4. 与论文 7B 结果的对比**

论文在 Qwen2.5-7B-1M（instruct model）上观察到"half of the heads exhibit highly sparse KV caches"。我们在 base model 上也看到类似 pattern（book 上 ~42% sparse），但 sparse 比例略低于论文的"half"。可能原因：(a) base vs instruct model 差异，(b) 数据规模/分布差异，(c) 7B 可能确实有更多 sparse heads。

### Step 3: 下一步判断

**结论：小模型有分化 → hybrid-head 方向值得继续**

既然 0.6B 就有 ~25% sparse + ~18% dense 的分化 pattern，那 hybrid-head architecture 在小模型上应该是可行的。之前小模型 hybrid-head 训练失败的原因可能不是"缺乏分化能力"，而是：
- 训练时间/方式不足以让分化涌现
- 初始化策略不当
- SW window size 的选择需要更仔细

**建议的后续实验**：
1. 用 Fast KVzip 发现的 sparse/dense pattern 直接指导 hybrid-head assignment（而不是随机分配）
2. 对比 informed assignment vs random assignment vs uniform full attention
3. 在 0.6B 规模上验证，因为成本最低且分化 pattern 已经存在

### 实验代码和数据位置

- Gate 训练结果：`result_gate/{model_name}/`
- Retention 原始数据：`experiments/results/retention_{model_name}.pt`
- Bias 原始数据：`experiments/results/bias_{model_name}.pt`
- 图表：`experiments/figs/{model_name}/` 和 `experiments/figs/cross_scale/`
- 脚本：`experiments/run_experiment.sh`（训练）、`experiments/run_retention.sh`（retention）、`experiments/run_analyze.sh`（出图）
