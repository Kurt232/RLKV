# RLKV Efficiency Evaluation Report

## Evaluation Setup

- **Model**: DeepSeek-R1-Distill-Llama-3.1-8B (Llama-3.1-8B-R1)
- **Benchmark**: MATH-500 (500 samples, max 8192 tokens)
- **Hardware**: Single NVIDIA A100 80GB GPU
- **Inference Engine**: SGLang v0.5.2 (custom fork with HeadRealloc backend)
- **Head Distribution**: `llama_lr1e-2_ep2_bs32_reg1e-3_tau0.5` (RL-trained)
- **Compression**: Sink=16, Window=64 for compressed heads

---

## 1. SGLang Integration: Challenges and Solutions

### 1.1 Dual KV Pool Architecture

**Problem**: SGLang's native KV cache assumes uniform pool — all heads share the same token capacity. RLKV requires per-head heterogeneous allocation: full-attention heads get full-length KV cache, while compressed heads only need a small circular buffer (sink + recent window).

**Solution**: Implemented `HeadReallocKVPool` — a dual-pool design that maintains two separate token pools:
- **Full pool**: standard paged KV cache for reasoning-critical heads (size = `max_total_num_tokens`)
- **Comp pool**: fixed-size circular buffer for compressed heads (size = `max_running_requests * window`)

The comp pool uses a circular overwrite strategy with `full_to_comp_mapping` — no allocation/eviction needed, completely transparent to SGLang's scheduler and radix cache.

### 1.2 Attention Backend

**Problem**: SGLang's FlashInfer attention backend assumes all heads read from the same KV index. With dual pools, full heads and comp heads need different index mappings at decode time.

**Solution**: Implemented `HeadReallocAttentionBackend` that:
- Prefill: runs full attention on all heads (comp pool not yet active)
- Decode: dispatches two separate FlashInfer calls — one for full heads (standard paging) and one for comp heads (circular buffer indices)
- Fused the two decode calls with custom Triton kernels to reduce kernel launch overhead

### 1.3 CUDA Graph Compatibility

**Problem**: SGLang aggressively uses CUDA graphs for decode batches. The dual-dispatch attention pattern with dynamic head masks was initially incompatible — CUDA graph capture requires fixed control flow and tensor shapes.

**Solution**: Pre-materialized head masks as static CUDA tensors during engine init. The dual FlashInfer calls operate on fixed-shape index buffers that are updated in-place, allowing full CUDA graph capture for both full and comp attention paths.

### 1.4 Memory Rebalancing

**Problem**: With sparsity *s*, *s* fraction of heads only need `window`-sized buffers instead of `max_total_num_tokens`. The freed memory should expand the full pool (more concurrent sequences), but SGLang's memory allocator computes pool size once at init.

**Solution**: Budget-based rebalancing at init time:
```
total_budget = max_total_num_tokens * total_heads
comp_cost    = max_running_requests * window * num_comp_heads
full_pool    = max((budget - comp_cost) / num_full_heads, max_total_num_tokens)
```
This allows the full pool to grow beyond the default `max_total_num_tokens`, directly translating memory savings into higher concurrency.

### 1.5 Comp Pool Sizing

**Problem**: The comp pool size depends on the maximum number of concurrent requests, which is a scheduler-level concept. Early designs exposed a custom `--rlkv-max-concurrent` parameter, creating coupling between the inference engine and RLKV-specific logic.

**Solution**: Reuse SGLang's native `--max-running-requests` parameter. The comp pool size is deterministically computed as `max_running_requests * window` inside `HeadReallocKVPool.__init__`, requiring zero changes to SGLang's CLI or scheduler.

---

## 2. Throughput Results

### 2.1 Main Results Table

| Method | Sparsity | Max Concurrent | Total Time (s) | Gen Throughput (tok/s) | Req Throughput (req/s) | Speedup |
|--------|----------|---------------|-----------------|----------------------|----------------------|---------|
| Full   | 0%       | 150           | 1036.3          | 1499.8               | 0.483                | 1.00x   |
| RLKV   | 20%      | 200           | 873.7           | 1757.8               | 0.572                | 1.19x   |
| RLKV   | 40%      | 250           | 665.5           | 2184.5               | 0.751                | 1.56x   |
| RLKV   | 50%      | 300           | 574.2           | 2552.8               | 0.871                | 1.80x   |
| RLKV   | 60%      | 375           | 504.2           | 2940.9               | 0.992                | 2.06x   |
| RLKV   | 80%      | 750           | 1085.1          | 3136.3               | 0.461                | 0.95x   |

> **Note**: Speedup is computed as `full_time / rlkv_time`. The sp=0.8 case shows degraded wall-clock time despite high token throughput, because excessive compression causes output length explosion (avg 6806 tokens vs 3108 for full), indicating quality degradation.

### 2.2 Accuracy Results

| Method | Sparsity | Accuracy (%) | Avg Output Len | Error Rate (%) | Incorrect (%) | Overlength (%) |
|--------|----------|-------------|-----------------|----------------|--------------|----------------|
| Full   | 0%       | 79.4        | 3108            | 20.6           | 5.4          | 15.2           |
| RLKV   | 20%      | 78.6        | 3072            | 21.4           | 5.4          | 16.0           |
| RLKV   | 40%      | 79.6        | 2908            | 20.4           | 7.2          | 13.2           |
| RLKV   | 50%      | 77.6        | 2932            | 22.4           | 10.2         | 12.2           |
| RLKV   | 60%      | 73.8        | 2965            | 26.2           | 14.4         | 11.8           |
| RLKV   | 80%      | 34.8        | 6806            | 65.2           | 4.8          | 60.4           |

### 2.3 Key Observations

1. **Near-lossless at sp=0.2-0.4**: Accuracy within 1% of full attention (78.6-79.6% vs 79.4%), with 1.19-1.56x throughput speedup.

2. **Sweet spot at sp=0.5-0.6**: 1.80-2.06x speedup with moderate accuracy trade-off (73.8-77.6%). The sp=0.6 configuration achieves >2x speedup.

3. **Collapse at sp=0.8**: Accuracy drops to 34.8% with 60.4% overlength rate — the model loses coherent reasoning and generates repetitive/degenerate outputs (avg 6806 tokens, 370/500 hit max length).

4. **Throughput scales with concurrency**: Token throughput increases monotonically with sparsity (1500 -> 3136 tok/s) because higher sparsity enables more concurrent requests. However, wall-clock time depends on both throughput and output quality.

---

## 3. Raw Stats

### 3.1 Token Statistics

| Config | Total Input | Total Output | Avg Input | Avg Output | Min Output | Max Output | Hit Max |
|--------|------------|-------------|-----------|------------|------------|------------|---------|
| Full        | 72,800 | 1,554,197 | 145.6 | 3,108 | 273  | 8,192 | 89  |
| RLKV sp=0.2 | 72,800 | 1,535,875 | 145.6 | 3,072 | 242  | 8,192 | 89  |
| RLKV sp=0.4 | 72,800 | 1,453,825 | 145.6 | 2,908 | 213  | 8,192 | 71  |
| RLKV sp=0.5 | 72,800 | 1,465,916 | 145.6 | 2,932 | 224  | 8,192 | 70  |
| RLKV sp=0.6 | 72,800 | 1,482,632 | 145.6 | 2,965 | 269  | 8,192 | 65  |
| RLKV sp=0.8 | 72,800 | 3,403,056 | 145.6 | 6,806 | 472  | 8,192 | 370 |

### 3.2 Latency

| Config | Total Time (s) | Avg Latency/Req (s) |
|--------|---------------|---------------------|
| Full        | 1036.25 | 2.07 |
| RLKV sp=0.2 | 873.73  | 1.75 |
| RLKV sp=0.4 | 665.52  | 1.33 |
| RLKV sp=0.5 | 574.24  | 1.15 |
| RLKV sp=0.6 | 504.15  | 1.01 |
| RLKV sp=0.8 | 1085.05 | 2.17 |
