# Optimization Ladder: From Upstream Baseline to Current Kernels

This report breaks the two biggest wins in this repo into cumulative optimization layers so the progression is easy to present:

- `causal_conv1d_py`
- `gated_deltanet_recompute_w_u_py`

It builds on [PRESENTATION_BASELINE_VS_OPTIMIZED.md](PRESENTATION_BASELINE_VS_OPTIMIZED.md), but instead of comparing only the first and last versions, it measures the intermediate stages that lead from the upstream reference kernel to the current implementation in this repo.

## Methodology

- Baseline source: the current upstream `submission.py` under [`gpu-mode/reference-kernels/main/problems/helion`](https://github.com/gpu-mode/reference-kernels/tree/main/problems/helion).
- Final source: the current local kernels in [`causal_conv1d_py/submission.py`](causal_conv1d_py/submission.py) and [`gated_deltanet_recompute_w_u_py/submission.py`](gated_deltanet_recompute_w_u_py/submission.py).
- Each stage is cumulative: stage `N` includes the changes from stages `0..N-1`.
- Benchmarks were run on March 17, 2026 on the `helion` H200 server with the official harness:
  - `python eval.py benchmark <kernel_dir>/`
- The staged kernels were generated into temporary benchmark directories, synced to the server, and benchmarked there so every stage used the same `task.py`, `task.yml`, and harness.
- Reported latency is the `min` time from the official benchmark output.
- Geomean speedup is computed across the three official benchmark shapes for each kernel.

## Benchmark Shapes

### `causal_conv1d_py`

| Benchmark | Shape |
|---|---|
| 0 | `B=1, D=1536, S=2048, W=4` |
| 1 | `B=1, D=2560, S=2048, W=4` |
| 2 | `B=1, D=2560, S=4096, W=4` |

### `gated_deltanet_recompute_w_u_py`

| Benchmark | Shape |
|---|---|
| 0 | `B=1, T=64, H=1, K=64, V=64` |
| 1 | `B=2, T=512, H=3, K=64, V=64` |
| 2 | `B=2, T=1024, H=3, K=64, V=64` |

## `causal_conv1d_py` Optimization Ladder

### Executive Summary

The causal conv kernel does not get fast by adding more math tricks. It gets fast by:

1. removing wrapper-side memory work,
2. making the kernel operate directly on the true output domain,
3. and then tuning the tile size so the memory-bound kernel launches far fewer, wider programs.

The biggest step is not arithmetic cleanup. It is the transition from the baseline launch/config regime to a config that matches the real benchmark shapes.

### Stage Table

| Stage | What changed | Bench 0 min | Bench 1 min | Bench 2 min | Geomean min | Incremental speedup | Cumulative speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Upstream baseline | `249.4 us` | `411.8 us` | `816.3 us` | `437.7 us` | `1.00x` | `1.00x` |
| 1 | Remove triple accumulator structure | `249.5 us` | `411.8 us` | `816.2 us` | `437.7 us` | `1.00x` | `1.00x` |
| 2 | Move causal padding logic into the kernel | `237.1 us` | `394.5 us` | `788.3 us` | `419.3 us` | `1.04x` | `1.04x` |
| 3 | Retune launch config for long-`S` benchmark shapes | `10.4 us` | `15.9 us` | `29.3 us` | `16.9 us` | `24.78x` | `25.86x` |
| 4 | Final repo kernel: benchmark/default split configs | `9.5 us` | `14.1 us` | `25.5 us` | `15.1 us` | `1.12x` | `29.06x` |

### What Each Stage Did

#### Stage 0: Upstream baseline

Source:

- [Upstream baseline `causal_conv1d_py/submission.py`](https://raw.githubusercontent.com/gpu-mode/reference-kernels/main/problems/helion/causal_conv1d_py/submission.py)

Important properties of the baseline:

- The wrapper does host-side causal padding with `torch.zeros(...)` plus `torch.cat(...)`.
- The kernel reads from the padded tensor instead of the original sequence tensor.
- The kernel performs three redundant accumulations and averages them.
- The launch config uses placeholder-scale tiles such as `block_sizes=[1, 8]`, `num_warps=1`, `num_stages=1`.

This version is dominated by two kinds of waste:

- extra memory traffic from building and reading the padded tensor,
- and tiny launch tiles that explode the number of programs on large `S`.

#### Stage 1: Remove the triple accumulator structure

Change:

- Collapse the three identical accumulators into one accumulator.

Why it was worth isolating:

- It is the cleanest arithmetic simplification in the baseline.
- It proves whether the baseline was compute-bound or dominated by something else.

Measured result:

- Essentially no change on H200.

Interpretation:

- The upstream baseline was not bottlenecked primarily by the duplicated arithmetic in isolation.
- The wrapper-side padding plus tiny launch geometry were still the dominant cost centers.

#### Stage 2: Move causal padding logic into the kernel

Change:

- Delete the wrapper-side `pad_zeros` + `torch.cat(...)`.
- Load directly from the original `x` tensor.
- Reconstruct causality inside the kernel with:
  - `src_idx = rs.index + j - (W - 1)`
  - `safe_idx = src_idx.clamp(min=0)`
  - a validity mask on negative source positions

Why it matters:

- It removes the extra allocation and concatenation from the wrapper.
- It eliminates reading a larger padded tensor.
- It makes the kernel operate over the true output length `S` rather than the padded length `L`.

Measured result:

- `1.04x` geomean improvement over stage 1.

Interpretation:

- This is a real but modest win.
- It reduces memory work, but the kernel is still running under a very small-tile config, so the launch regime remains the bigger bottleneck.

#### Stage 3: Retune the launch config for the actual benchmark shapes

Change:

- Keep the stage-2 algorithm.
- Replace the placeholder config with a tuned config:
  - `block_sizes=[1, 256]`
  - `num_warps=2`
  - `num_stages=1`

Why it matters:

- This kernel is memory-bound and has a very small filter width (`W=4`).
- Larger `S` tiles drastically reduce the number of launches on the official long-sequence shapes.
- Wider `S` blocks also let adjacent output positions reuse overlapping input windows more effectively.

Measured result:

- `24.78x` incremental geomean speedup over stage 2.

Interpretation:

- This is the main turning point.
- The arithmetic simplifications and boundary cleanup help, but the decisive gain comes from matching the launch geometry to the real workload.

#### Stage 4: Final repo kernel

Current code:

- Config split: [`causal_conv1d_py/submission.py`](causal_conv1d_py/submission.py#L11)
- Hot loop: [`causal_conv1d_py/submission.py`](causal_conv1d_py/submission.py#L44)
- Direct dispatch without wrapper padding: [`causal_conv1d_py/submission.py`](causal_conv1d_py/submission.py#L65)

Final changes relative to stage 3:

- Use a benchmark-specific config:
  - `_BENCH = helion.Config(block_sizes=[1, 512], num_warps=2, num_stages=1)`
- Keep a separate default config for smaller test shapes:
  - `_DEFAULT = helion.Config(block_sizes=[1, 256], num_warps=4, num_stages=1)`
- Preserve the in-kernel clamp-and-mask causal logic.
- Preserve direct dispatch on the unpadded input.

Measured result:

- `1.12x` incremental geomean speedup over stage 3.
- `29.06x` cumulative geomean speedup over the upstream baseline.

### Final Kernel Anatomy

The current causal conv kernel is fast because it combines four ideas cleanly:

1. `custom_kernel` dispatches directly on the original tensor and avoids wrapper-side padding.
2. The inner loop uses in-kernel causal addressing and masking instead of a prebuilt padded tensor.
3. `hl.specialize(w.size(1))` fixes `W` at compile time so the short inner loop is easy to optimize.
4. The benchmark shapes use much larger `S` tiles than the upstream baseline, which is the main reason latency collapses from hundreds of microseconds to low double digits.

## `gated_deltanet_recompute_w_u_py` Optimization Ladder

### Executive Summary

The recompute kernel gets faster in two large jumps:

1. first by deleting the baseline's reverse duplicate pass,
2. then by replacing explicit outer-product accumulation with direct `hl.dot` matmuls.

After that, the remaining gains come from making the matmul version friendlier to the compiler and to the GPU register/cache hierarchy.

### Stage Table

| Stage | What changed | Bench 0 min | Bench 1 min | Bench 2 min | Geomean min | Incremental speedup | Cumulative speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Upstream baseline | `368.8 us` | `359.6 us` | `359.1 us` | `362.5 us` | `1.00x` | `1.00x` |
| 1 | Remove reverse duplicate pass and averaging | `83.0 us` | `91.7 us` | `91.3 us` | `88.6 us` | `4.09x` | `4.09x` |
| 2 | Replace scalar outer-product loops with direct chunk matmuls | `9.6 us` | `11.8 us` | `12.6 us` | `11.3 us` | `7.87x` | `32.19x` |
| 3 | Add register tiling and a matmul-friendly config | `5.6 us` | `6.2 us` | `7.0 us` | `6.2 us` | `1.80x` | `58.08x` |
| 4 | Final repo kernel: per-shape submission config hardening | `5.6 us` | `6.3 us` | `7.0 us` | `6.3 us` | `~1.00x (tied)` | `57.77x` |

### What Each Stage Did

#### Stage 0: Upstream baseline

Source:

- [Upstream baseline `gated_deltanet_recompute_w_u_py/submission.py`](https://raw.githubusercontent.com/gpu-mode/reference-kernels/main/problems/helion/gated_deltanet_recompute_w_u_py/submission.py)

Important properties of the baseline:

- It computes the same logical transform twice:
  - one forward accumulation pass,
  - one reverse accumulation pass,
  - then averages the results.
- The kernel builds the result through explicit per-column outer-product style updates:
  - `a_col[:, None] * vector[None, :]`
- It uses placeholder launch configs (`num_warps=1`, `num_stages=1`).

This gives correct behavior, but it is structurally far from the hardware's preferred execution pattern for dense chunk-local linear algebra.

#### Stage 1: Remove the reverse duplicate pass

Change:

- Keep the explicit accumulation structure.
- Delete the second reverse traversal and final averaging.

Why it matters:

- The baseline was doing nearly double the work in the hottest part of the kernel.
- This change isolates how much of the runtime came from duplicated math before any more aggressive reformulation.

Measured result:

- `4.09x` geomean speedup over the baseline.

Interpretation:

- The duplicate pass was a very large cost center.
- Even before changing the algorithmic form, simply deleting redundant work collapses latency from roughly `362 us` geomean to roughly `89 us`.

#### Stage 2: Replace scalar outer-product accumulation with direct chunk matmuls

Change:

- Stop building `w` and `u` through explicit `for ci in range(C)` outer-product updates.
- Form scaled chunk views:
  - `k_scaled = k_chunk * (beta_vals * exp(g_vals))[:, None]`
  - `v_scaled = v_chunk * beta_vals[:, None]`
- Compute the outputs directly as:
  - `hl.dot(a_chunk, k_scaled)`
  - `hl.dot(a_chunk, v_scaled)`

Why it matters:

- This matches the real math of the kernel directly:
  - `w = A @ (k * beta * exp(g))`
  - `u = A @ (v * beta)`
- It converts the hot path from many tiny scalar-style outer-product updates into dense chunk-local GEMM-style work.
- That lets the implementation map onto Helion's dot path instead of emulating matmul structure manually.

Measured result:

- `7.87x` incremental geomean speedup over stage 1.
- `32.19x` cumulative speedup over the baseline.

Interpretation:

- This is the biggest algorithmic win in the entire recompute ladder.
- It is the moment the kernel stops behaving like a scalar accumulation loop and starts behaving like a real matmul kernel.

#### Stage 3: Add register tiling and a matmul-friendly config

Change:

- Preserve the direct chunk matmul formulation.
- Add:
  - `block_k = hl.register_block_size(K)`
  - `block_v = hl.register_block_size(V)`
- Tile the output feature dimensions in register-friendly blocks.
- Use a matmul-oriented config:
  - `block_sizes=[64, 64]`
  - `num_warps=4`
  - `num_stages=1`

Why it matters:

- The matmul reformulation unlocked the right computation pattern.
- This stage improves how that pattern is scheduled and materialized.
- Register tiling reduces pressure on the hottest `K` and `V` output loops and helps the dot path operate on cleaner local tiles.

Measured result:

- `1.80x` incremental geomean speedup over stage 2.
- `58.08x` cumulative speedup over the baseline.

Interpretation:

- Once the kernel is in the right algorithmic form, this is the major remaining structural performance layer.
- It turns the direct matmul rewrite into a submission-class fast kernel.

#### Stage 4: Final repo kernel

Current code:

- Per-shape configs: [`gated_deltanet_recompute_w_u_py/submission.py`](gated_deltanet_recompute_w_u_py/submission.py#L11)
- Specialization and register tiling: [`gated_deltanet_recompute_w_u_py/submission.py`](gated_deltanet_recompute_w_u_py/submission.py#L35)
- Fused scaled-dot hot path: [`gated_deltanet_recompute_w_u_py/submission.py`](gated_deltanet_recompute_w_u_py/submission.py#L47)

Final changes relative to stage 3:

- Replace the one-size-fits-all config with per-shape hardcoded configs.
- Increase `num_stages` on larger shapes.
- Increase `num_warps` to `8` for the larger benchmark shapes while preserving the direct-layout dot formulation.

Measured result:

- Ties stage 3 on benchmarks 0 and 2.
- Is marginally slower on benchmark 1 on this H200 run (`6.2 us` -> `6.3 us`).
- The geomean difference versus stage 3 is about `0.5%`, so for presentation purposes this is best treated as a tie rather than a new performance step.
- Ends at `57.77x` cumulative geomean speedup versus the upstream baseline.

Interpretation:

- Stage 4 is best thought of as the submission-hardened final kernel rather than a pure speed-only layer.
- It is the version we keep in the repo because it is the real multi-shape submission implementation.
- The optimization story effectively stops at stage 3.
- Stage 4 should be presented as "final submitted kernel, performance tied with stage 3 on H200" rather than as an additional optimization win.

### Final Kernel Anatomy

The current recompute kernel is fast because it combines five ideas:

1. It expresses the actual WY recomputation as direct chunk-local matrix products.
2. It fuses the `beta` and `exp(g)` scaling into the dot inputs instead of materializing extra intermediates.
3. It specializes `C`, `K`, and `V`, which gives the compiler fixed hot-path dimensions.
4. It tiles `K` and `V` with `hl.register_block_size(...)`, making the dot path friendlier to register reuse.
5. It uses hardcoded per-shape configs so the submission can stay fast across both test and benchmark shapes.

## Key Takeaways for the Presentation

### `causal_conv1d_py`

- The arithmetic cleanup was not the main win.
- Moving the causal boundary logic into the kernel helped a little.
- The dominant gain came from tuning the launch geometry for the actual long-sequence benchmark shapes.
- Final outcome: `29.06x` cumulative geomean speedup.

### `gated_deltanet_recompute_w_u_py`

- The first big win came from deleting duplicate work.
- The second and biggest win came from changing the kernel from explicit scalar-style accumulation into direct matmul structure.
- Register tiling added another meaningful step once the algorithm was in the right form.
- Final outcome: `57.77x` cumulative geomean speedup.

## Recommended Slide Framing

If this needs to fit into a short talk, the cleanest framing is:

1. Baseline: correct but structurally mismatched to the hardware.
2. Remove obvious waste.
3. Rewrite the hot path to match the real computation.
4. Tune launch geometry and shape-specific configs.
5. Land the submission-safe final kernel.

That framing matches both kernels, even though the exact weights differ:

- `causal_conv1d_py` is primarily a memory-traffic and launch-geometry story.
- `gated_deltanet_recompute_w_u_py` is primarily an algorithm-shape and matmul-mapping story.
