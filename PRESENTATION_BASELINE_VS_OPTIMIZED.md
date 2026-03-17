# Baseline vs Optimized Kernel Analysis

This note compares the current optimized kernels in this repository against the
current upstream baselines from:

- https://github.com/gpu-mode/reference-kernels/tree/main/problems/helion

The focus here is on the two biggest presentation-worthy wins:

- `causal_conv1d_py`
- `gated_deltanet_recompute_w_u_py`

Benchmarks were run on the `helion` H200 VM with the same `eval.py benchmark`
driver for both baseline and optimized kernels.

## Executive Summary

| Kernel | Baseline Fastest Min | Optimized Fastest Min | Speedup |
|---|---:|---:|---:|
| `causal_conv1d_py` | `249.5 us` | `9.5 us` | `26.3x` |
| `gated_deltanet_recompute_w_u_py` | `359.0 us` | `5.6 us` | `64.1x` |

## Per-Shape Benchmarks

### `causal_conv1d_py`

| Benchmark Shape | Baseline Min | Optimized Min | Speedup |
|---|---:|---:|---:|
| `{'B': 1, 'D': 1536, 'S': 2048, 'W': 4}` | `249.5 us` | `9.5 us` | `26.3x` |
| `{'B': 1, 'D': 2560, 'S': 2048, 'W': 4}` | `411.8 us` | `14.2 us` | `29.0x` |
| `{'B': 1, 'D': 2560, 'S': 4096, 'W': 4}` | `816.7 us` | `25.5 us` | `32.0x` |

Geomean speedup across the 3 benchmark shapes: `29.0x`

### `gated_deltanet_recompute_w_u_py`

| Benchmark Shape | Baseline Min | Optimized Min | Speedup |
|---|---:|---:|---:|
| `{'B': 1, 'T': 64, 'H': 1, 'K': 64, 'V': 64}` | `368.9 us` | `5.6 us` | `65.9x` |
| `{'B': 2, 'T': 512, 'H': 3, 'K': 64, 'V': 64}` | `359.3 us` | `6.3 us` | `57.0x` |
| `{'B': 2, 'T': 1024, 'H': 3, 'K': 64, 'V': 64}` | `359.0 us` | `7.0 us` | `51.3x` |

Geomean speedup across the 3 benchmark shapes: `57.8x`

## `causal_conv1d_py` Deep Analysis

### What the baseline does

The upstream baseline is intentionally inefficient. It has three main
performance problems:

1. It pads on the host before launching the kernel.
   - In baseline `custom_kernel`, it creates `pad_zeros` and concatenates it
     with the input tensor before dispatch.
   - That introduces an extra allocation, an extra write, and extra memory
     traffic before the GPU kernel even starts.

2. It computes the same convolution three times.
   - The baseline kernel keeps `acc1`, `acc2`, and `acc3`.
   - Inside the inner loop, it reloads the same activation and the same weight
     three separate times.
   - It then averages the three identical accumulators.
   - This is pure redundant compute and redundant memory traffic.

3. Its configs are placeholders, not tuned benchmark configs.
   - The baseline uses `block_sizes=[1, 8], num_warps=1, num_stages=1` almost
     everywhere.
   - Those settings leave throughput on the table on larger benchmark shapes.

### What the optimized kernel does instead

The optimized implementation is in
[/Users/alhinai/Desktop/helion/causal_conv1d_py/submission.py](/Users/alhinai/Desktop/helion/causal_conv1d_py/submission.py#L11).

Key changes:

1. Removed host-side padding completely.
   - The optimized `custom_kernel` passes the original input directly to the
     kernel.
   - There is no `torch.zeros(...)` allocation and no `torch.cat(...)`.

2. Moved causal boundary handling into the kernel.
   - In the kernel body, the source index is computed as:
     - `src_idx = rs.index + j - (W - 1)`
   - Then it uses:
     - `safe_idx = src_idx.clamp(min=0)`
     - `valid = (src_idx >= 0).to(torch.float32)`
   - This gives correct causal behavior without materializing padded input.

3. Replaced triple accumulation with a single accumulation.
   - The optimized kernel uses one accumulator:
     - `acc = hl.zeros([rd, rs], dtype=torch.float32)`
   - For each filter tap, it loads the input once and the coefficient once.
   - That alone removes roughly `3x` of the baseline inner-loop work.

4. Reduced repeated memory loads.
   - Baseline inner loop:
     - loads `x_pad[..., rs.index + j]` three times
     - loads `w[..., j]` three times
   - Optimized inner loop:
     - loads `x[...]` once
     - loads `w[...]` once
   - This matters because Conv1D here is memory-bound.

5. Tuned configs for benchmark shapes.
   - Benchmark path:
     - `_BENCH = helion.Config(block_sizes=[1, 512], num_warps=2, num_stages=1)`
   - Default/test path:
     - `_DEFAULT = helion.Config(block_sizes=[1, 256], num_warps=4, num_stages=1)`
   - On the H200 benchmark shapes, the lower-warp, larger-S tile benchmark
     path won clearly.

### Why these changes help so much

There are three stacked wins:

1. Less work before the kernel.
   - No host padding.

2. Less work inside the kernel.
   - No triple accumulation.
   - No triple reload of activations and coefficients.

3. Better GPU launch geometry.
   - Much larger sequence tile.
   - Better occupancy/throughput balance for the actual benchmark regime.

### Code-level mapping

Optimized code highlights:

- Config split for benchmark shapes:
  [causal_conv1d_py/submission.py:11](/Users/alhinai/Desktop/helion/causal_conv1d_py/submission.py#L11)
- Shape dispatch:
  [causal_conv1d_py/submission.py:16](/Users/alhinai/Desktop/helion/causal_conv1d_py/submission.py#L16)
- In-kernel causal clamp/mask:
  [causal_conv1d_py/submission.py:47](/Users/alhinai/Desktop/helion/causal_conv1d_py/submission.py#L47)
- Single accumulation and bias add:
  [causal_conv1d_py/submission.py:46](/Users/alhinai/Desktop/helion/causal_conv1d_py/submission.py#L46)
  [causal_conv1d_py/submission.py:54](/Users/alhinai/Desktop/helion/causal_conv1d_py/submission.py#L54)

### Presentation takeaway

`causal_conv1d_py` is a strong example of a kernel getting dramatically faster
not by adding exotic complexity, but by removing waste:

- remove host padding
- remove duplicate math
- reduce memory traffic
- tune the launch to the real benchmark shapes

That combination turned an intentionally slow teaching baseline into a
production-like memory-bound kernel with a `26x+` fastest-shape speedup.

## `gated_deltanet_recompute_w_u_py` Deep Analysis

### What the baseline does

The upstream baseline is also intentionally inefficient. Its main performance
issues are structural:

1. It computes the transform with explicit elementwise accumulation loops.
   - For each chunk, it iterates over `ci in range(C)` and manually expands:
     - `a_col[:, None] * vector[None, :]`
   - This is much less GPU-friendly than a dense matrix contraction.

2. It performs the whole computation twice.
   - First forward:
     - `w_acc1`, `u_acc1`
   - Then backward:
     - `w_acc2`, `u_acc2`
   - It averages the two at the end.

3. It maintains four large accumulators per tile.
   - `w_acc1`, `u_acc1`, `w_acc2`, `u_acc2`
   - That inflates data movement and on-chip state pressure.

4. Its configs are placeholders.
   - Baseline uses `num_warps=1, num_stages=1` across shapes.

### What the optimized kernel does instead

The optimized implementation is in
[/Users/alhinai/Desktop/helion/gated_deltanet_recompute_w_u_py/submission.py](/Users/alhinai/Desktop/helion/gated_deltanet_recompute_w_u_py/submission.py#L11).

Key changes:

1. Reframed the problem as direct chunk-local matmuls.
   - The target math is:
     - `w = A @ (k * beta * exp(g))`
     - `u = A @ (v * beta)`
   - The optimized kernel computes exactly those contractions directly with
     `hl.dot`.

2. Removed the reverse-pass duplicate accumulation.
   - The baseline does two passes and averages them.
   - The optimized kernel does one mathematically direct pass.

3. Moved from scalar/broadcast accumulation to dense contractions.
   - Instead of manually doing repeated outer-product style updates column by
     column, the optimized kernel calls:
     - `hl.dot(a_chunk, scaled_k_tile, out_dtype=torch.float32)`
     - `hl.dot(a_chunk, scaled_v_tile, out_dtype=torch.float32)`
   - This is far better aligned with GPU hardware.

4. Used chunk-local direct layout.
   - The kernel tiles over `[B, T, H]` with `block_size=[1, C, 1]`.
   - Each program works on one chunk matrix at a time.

5. Specialized the hot dimensions.
   - `C = hl.specialize(A.shape[-1])`
   - `K = hl.specialize(K)`
   - `V = hl.specialize(V)`
   - This gives the compiler fixed sizes for the dense path.

6. Added register-sized inner tiling.
   - `block_k = hl.register_block_size(K)`
   - `block_v = hl.register_block_size(V)`
   - Then the kernel iterates over `tile_k` and `tile_v` explicitly.

7. Fused scaling into the matmul inputs.
   - `beta_vals` and `gate_vals` are formed once per chunk tile.
   - The `k` and `v` operands are scaled before the dot product rather than
     being accumulated elementwise across repeated loop bodies.

8. Tuned per-shape configs.
   - Small/test shapes:
     - `block_sizes=[64, 64], num_warps=4, num_stages=1/2`
   - Benchmark and larger shapes:
     - `block_sizes=[64, 64], num_warps=8, num_stages=2/3`
   - That is a major improvement over the placeholder `num_warps=1`.

### Why these changes help so much

This kernel gets its speedup from both algorithmic form and config quality.

1. Fewer passes over the chunk.
   - The baseline does forward accumulation and backward accumulation.
   - The optimized kernel computes the intended contractions once.

2. Better math mapping.
   - Dense `hl.dot` is a much better fit for the GPU than repeated
     broadcast-based column accumulation.

3. Lower accumulator overhead.
   - The baseline maintains four full output accumulators.
   - The optimized path emits the two actual outputs directly.

4. Better on-chip locality.
   - Chunk-local tiling and register-sized K/V sub-tiles reduce unnecessary
     traffic and help the compiler keep the hot path compact.

5. Much better shape-specific launch parameters.
   - The baseline is essentially untuned.
   - The optimized kernel uses tuned warp/stage combinations that scale with
     sequence length and feature size.

### Code-level mapping

Optimized code highlights:

- Per-shape tuned configs:
  [gated_deltanet_recompute_w_u_py/submission.py:11](/Users/alhinai/Desktop/helion/gated_deltanet_recompute_w_u_py/submission.py#L11)
- Chunk-local tiling and specialization:
  [gated_deltanet_recompute_w_u_py/submission.py:35](/Users/alhinai/Desktop/helion/gated_deltanet_recompute_w_u_py/submission.py#L35)
- Register-sized K/V tiling:
  [gated_deltanet_recompute_w_u_py/submission.py:44](/Users/alhinai/Desktop/helion/gated_deltanet_recompute_w_u_py/submission.py#L44)
- Fused scaling:
  [gated_deltanet_recompute_w_u_py/submission.py:50](/Users/alhinai/Desktop/helion/gated_deltanet_recompute_w_u_py/submission.py#L50)
- Direct `hl.dot` output path for `w`:
  [gated_deltanet_recompute_w_u_py/submission.py:54](/Users/alhinai/Desktop/helion/gated_deltanet_recompute_w_u_py/submission.py#L54)
- Direct `hl.dot` output path for `u`:
  [gated_deltanet_recompute_w_u_py/submission.py:61](/Users/alhinai/Desktop/helion/gated_deltanet_recompute_w_u_py/submission.py#L61)

### Presentation takeaway

`gated_deltanet_recompute_w_u_py` is the clearest example in this repo of
turning a pedagogical baseline into a proper GPU kernel:

- remove duplicate passes
- stop doing manual broadcast accumulation
- express the real computation as dense chunk-local matmuls
- specialize the hot dimensions
- tune the launch per shape

That shift is why this kernel shows the biggest result here: roughly `58x`
geomean speedup and `64x` fastest-shape speedup.

## High-Level Story for Slides

If you want a short narrative for presentation slides, this is the cleanest
version:

1. The upstream baselines were intentionally simple and intentionally slow.
2. The optimization work focused on removing wasted work first.
3. After that, the kernels were remapped to the GPU more naturally.
4. Finally, the launch configs were tuned to the benchmark shapes.

For these two kernels, the biggest wins came from:

- avoiding unnecessary memory movement
- deleting redundant math
- replacing scalar/broadcast-style accumulation with direct dense math
- specializing the implementation around the real benchmark regime

