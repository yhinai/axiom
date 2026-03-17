# External Kernel Review

## Scope

- Collected 66 public non-FP8 `submission.py` files under `external_submissions/`.
- Included direct code links from sources such as `DESU-CLUB`, `brandonin`, `Mistobaan/reference-kernels`, `svdrecbd`, `KG2468`, `ramizik`, `Ayush10`, `rajk97`, `andrewbriand`, `dpiresearch`, `InServiceOfX`, `gtcha2`, `gretchenboria`, and `rdspring1`.
- Some user-provided sources were only repo roots or PR pages without a direct `submission.py` path in the prompt, so they were not materialized in this first pass.

## High-confidence results

- `causal_conv1d_py`: a targeted sweep around the `rajk97`-style low-warp benchmark neighborhood found a better benchmark path.
- Change:
  - switched the benchmark config to `block_sizes=[1, 512], num_warps=2, num_stages=1`
  - switched the kernel body to clamp-and-mask boundary handling
- H200 validation on `helion`:
  - previous repo kernel: `0.0103 / 0.0149 / 0.0266 ms` min times
  - promoted kernel: `0.0094 / 0.0142 / 0.0255 ms` min times
  - correctness: passed all test shapes
- `gated_deltanet_chunk_fwd_o_py`: adopted the fused-accumulator pattern seen in the strongest external variants, especially `ramizik`.
- Change: compute the local `hl.dot(qk, v)` result into an accumulator and fuse the inter-chunk `hl.dot(q_g, h, acc=acc)` into the same accumulator.
- H200 validation on `helion`:
  - baseline: `0.0086 / 0.0097 / 0.0106 ms`
  - fused-acc variant: `0.0085 / 0.0096 / 0.0106 ms`
  - correctness: passed all test shapes
- `gated_deltanet_chunk_fwd_o_py`: a follow-up config sweep on the fused kernel found `num_warps=4` to be a clear winner on the H200.
- H200 validation on `helion` after promoting `num_warps=4`:
  - previous repo kernel: `0.0085 / 0.0096 / 0.0106 ms`
  - `num_warps=4`: `0.0052 / 0.0062 / 0.0072 ms` min times
  - correctness: passed all test shapes

## Rejected in first pass

- `gated_deltanet_recompute_w_u_py`: replacing `torch.exp(g)` with `torch.exp2(g * LOG2E)` did not help on the H200.
- H200 benchmark:
  - baseline: `0.0080 / 0.0095 / 0.0157 ms`
  - `exp2` variant: `0.0081 / 0.0096 / 0.0158 ms`

## Kernel-by-kernel assessment

### causal_conv1d_py

- Most external entries repeat the same core idea we already use: no host-side padding, specialize on `W`, and keep the kernel memory-bound and simple.
- The main differences are config-level:
  - `rajk97` uses `block_ptr`/pointer mixes with `block_sizes=[1, 512]` and `num_warps=1` on large benchmark shapes.
  - `brandonin`, `ramizik`, and `desu_club` lean on autotuned config metadata and ACFs.
- No obvious algorithmic rewrite stood out as strictly better than the current kernel.
- Best next step is a targeted config sweep on benchmark shapes only.

### gated_deltanet_chunk_fwd_h_py

- Our current kernel already includes the best math-side ideas found repeatedly in strong external variants:
  - recurrence kept in-kernel
  - `hl.dot(..., acc=state)` style update
  - exponential gating folded into the recurrence update
  - `exp2` rewrite
- Remaining upside appears to be config-only:
  - V tile size
  - number of stages
  - pointer vs tensor-descriptor mixes
  - possible ACF-backed configs on B200
- Targeted H200 sweeps did not produce an official-eval win over the current kernel.

### gated_deltanet_chunk_fwd_o_py

- This was the clearest external win area.
- Repeated strong patterns:
  - `exp2` for gating
  - fused `acc=` accumulation for local + global outputs
  - more aggressive persistent scheduling in some submissions
- We adopted the fused accumulation because it was low risk and benchmark-positive on the H200.
- We then used an H200-safe sweep harness to test the post-fusion config neighborhood and found that `num_warps=4` was dramatically better than the previous `num_warps=16` setting on the official benchmark shapes.
- Next upside is likely in config sweeps:
  - pointer-only indexing
  - persistent/interleaved pid types
  - broader multi-shape sweeps once the 4-warp baseline is locked in

### gated_deltanet_recompute_w_u_py

- The external field splits into two families:
  - direct matmul kernels like our current code, `ramizik`, and `brandonin`
  - tiled K/V-loop kernels like `ankitmaloo`
- The direct-matmul family is still the stronger fit for our current kernel shape and benchmark profile.
- The `exp2` rewrite alone was not a win on H200.
- Targeted H200 sweeps found some synthetic winners, but none of the top candidates beat the current kernel under the official `eval.py benchmark` harness.
- Most likely remaining upside is config-space rather than algebra:
  - `persistent_blocked` vs `persistent_interleaved`
  - smaller warp counts than our current config
  - mixed pointer/tensor-descriptor indexing

## Best next experiments

1. `gated_deltanet_recompute_w_u_py` deeper multi-shape search around flat/persistent hybrids if we want to keep pushing that kernel.
2. `gated_deltanet_chunk_fwd_h_py` only if we expand the search space beyond the safe config neighborhood already tested.
3. Broader multi-shape validation on a B200 box, because the H200-compatible remote copies still need `range_warp_specializes` stripped.
