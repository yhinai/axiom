# Helion Hackathon — GPU MODE (B200)

Optimized GPU kernels for the Helion DSL hackathon targeting NVIDIA B200 (Blackwell) GPUs.

## Results

| Kernel | Geomean | Rank | Key Speedup |
|--------|---------|------|-------------|
| Chunk Fwd H | 8.2 µs | Submitted | 1.6x over baseline |
| Recompute W/U | 55.3 µs | Submitted | 2.4x over baseline |
| Chunk Fwd O | — | Failed leaderboard | 1.7x over baseline |
| Causal Conv1D | ~15 µs | Deadline missed | 1.9x over baseline |
| FP8 Quant | — | Deadline missed | Correctness only |

## Optimizations by Kernel

### 1. Chunk Fwd H (Inter-Chunk State Recurrence)

Sequential state update: `h = h * decay + k^T @ v_gated` across chunks.

- **TF32 dot precision** — `dot_precision="tf32"` enables tensor cores for the two `[64,64]×[64,8]` dots per chunk. ~1.5x faster than IEEE, well within 1e-2 tolerance.
- **exp2 instead of exp** — `exp(x) = exp2(x * log2e)` maps to a single hardware instruction (`ex2`), avoiding the iterative refinement that `exp` requires.
- **Gate diff, not k** — `k^T @ (diff * α) = (k*α)^T @ diff` mathematically, but multiplying `diff [64,8]` is 8x fewer ops than multiplying `k [64,64]`.
- **Fused dot-accumulate** — `hl.dot(k.T, diff_gated, acc=state)` fuses the add into the dot, eliminating a separate read-modify-write.
- **Inner loop pipelining** — `range_num_stages=[0, 3]` prefetches next chunk's data while computing current chunk. Single most impactful config knob (~1.3x).
- **Per-shape warps** — `num_warps=16` for the tiny BH=1 shape (only 8 blocks on 148 SMs), `num_warps=4` for larger shapes.

### 2. Chunk Fwd O (Output Computation)

Combines local causal attention with global state: `out = scale * (qk_causal @ v + q_gated @ h)`.

- **TF32 dots + exp2 gating** — same as chunk_fwd_h, applied to 4 dot products per block.
- **Single-pass structure** — computes local attention and global state contribution without redundant intermediate values.
- **High warp count** — `num_warps=16` for compute-bound blocks with 4 dot products each.

### 3. Recompute W/U (WY Transform)

Recomputes `w = A @ (k * β * exp(g))` and `u = A @ (v * β)`.

- **Matmul reformulation (2.4x)** — replaced O(C²) element-wise loops with two `hl.dot(A, k_scaled)` calls, mapping directly to tensor core GEMMs. Largest single optimization across all kernels.
- **Persistent blocked kernel** — `pid_type='persistent_blocked'` with `num_sm_multiplier=16` keeps one program per SM looping over tiles, eliminating launch overhead.
- **Aggressive tuning** — `maxnreg=32` (increases occupancy), `num_warps=32`, `range_unroll_factors=[4]`, `l2_groupings=[16]` (improves cache locality).

### 4. Causal Conv1D (Depthwise Convolution)

`out[b,d,t] = bias[d] + Σ weight[d,k] * x[b,d,t-W+1+k]` — purely memory-bound.

- **Large S blocks** — `block_sizes=[1, 4096]` (up from 1024). Reduces launch count 4x and exploits 75% input overlap between adjacent positions (W=4). ~1.9x on shape 1.
- **Loop order** — `loop_orders=[[0, 2, 1]]` processes S (contiguous) before D, ensuring coalesced memory access.
- **L2 grouping** — `l2_groupings=[8]` groups adjacent blocks for cache reuse.
- **Compile-time unroll** — `hl.specialize(W)` makes the W=4 inner loop a compile-time constant, fully unrolled.

### 5. FP8 Quantization

Per-group absmax → scale → quantize.

- **Per-shape block sizes** — small shapes use small blocks (more parallelism), large shapes use large blocks (less overhead) with higher `num_warps`.
- **Specialized group size** — `hl.specialize(ncols)` enables fixed-width loads for group_size=128.

## Cross-Cutting Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| `static_shapes=True` | All kernels | Compile-time shape specialization, constant folding |
| `hl.specialize()` | All kernels | Makes dimensions compile-time constants |
| Per-shape `SHAPE_CONFIGS` | All kernels | Different sizes need different block/warp configs |
| `tensor_descriptor` indexing | Dot-heavy kernels | Uses B200's TMA hardware for bulk loads |
| `pointer` indexing | Elementwise kernels | Lower overhead for simple access patterns |

## Tools

- `eval.py` — test correctness + benchmark (`python eval.py test|benchmark|both <kernel>/`)
- `autotune_deltanet.py` — manual config sweep for all 3 deltanet kernels
- `tune_fwd_h.py` — Helion autotuner wrapper for chunk_fwd_h
- `leaderboard-tui/` — Go TUI for watching rankings in real-time
