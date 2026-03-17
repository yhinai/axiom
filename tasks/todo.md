- [x] Collect the public non-FP8 Helion submission files listed by the user into a single local folder.
- [x] Record source URLs and any fetch limitations for repos, PRs, or missing files.
- [x] Compare each external kernel against the current local `submission.py` implementations.
- [x] Identify optimizations that look plausibly beneficial on our H200/B200-style shapes.
- [x] Implement the highest-confidence optimizations in the local kernels.
- [x] Run correctness and benchmark validation for changed kernels locally and on `helion`.
- [x] Launch longer sweep experiments only for kernels/config spaces that still have upside.
- [x] Summarize adopted wins, rejected ideas, and next experiments.
- [x] Validate promising external `gated_deltanet_recompute_w_u_py` rewrites under the official `eval.py` harness.
- [x] Re-benchmark external `gated_deltanet_chunk_fwd_h_py` variants and reject the ones that do not beat the current kernel.

## Review

- Promoted the direct-layout `gated_deltanet_recompute_w_u_py` rewrite after it passed correctness and improved H200 benchmark mins from `8.0 / 9.5 / 15.7 us` to `5.6 / 6.3 / 6.9 us`.
- Reconfirmed the earlier wins in `causal_conv1d_py` and `gated_deltanet_chunk_fwd_o_py` with a fresh full four-kernel run on `helion`.
- Rejected deeper `gated_deltanet_chunk_fwd_h_py` external variants because the H200-safe runs were still slower than the current repo kernel, and one B200-tuned variant still relied on unsupported `range_warp_specializes` settings on H200.
