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

## Next Audit

- [x] Search current public Helion submissions and upstream PRs for per-kernel compute, memory, and buffering optimizations that are not yet reflected locally.
- [x] Compare each online optimization pattern against the current local kernels and note any gaps in `EXTERNAL_KERNEL_REVIEW.md`.
- [x] Implement any additional high-confidence wins and validate them with `eval.py test|benchmark` on `helion`.
- [x] Re-run the full all-kernels benchmark and record the best verified results.

## Online Audit Review

- Cross-checked the local kernels against current public submissions and upstream Helion PR descriptions for `causal_conv1d`, `gated_deltanet_chunk_fwd_h`, `gated_deltanet_chunk_fwd_o`, and `gated_deltanet_recompute_w_u`.
- Confirmed that the main publicly repeated wins are already present locally: clamp-and-mask causal loading, fused `chunk_fwd_o` accumulation, in-kernel recurrent `chunk_fwd_h` state updates, and the direct-layout `recompute_w_u` matmul rewrite.
- Benchmarked one remaining public `chunk_fwd_o` config family based on persistent/block-pointer scheduling and rejected it because it produced `nan`/correctness failures on all three official benchmark shapes on `helion`.
- Tried the last plausible causal-conv micro-optimization from public code review and rejected it because it introduced an invalid CPU/GPU mixed expression during Helion tracing rather than a viable faster kernel.

## Optimization Ladder

- [ ] Build cumulative optimization ladders for `causal_conv1d_py` and `gated_deltanet_recompute_w_u_py` from upstream baseline to current final kernel.
- [ ] Benchmark each ladder step on `helion` with the official `eval.py benchmark` harness.
- [ ] Compute per-step latency deltas and cumulative speedups for presentation use.
- [ ] Write a presentation-grade markdown report explaining each optimization step and its measured gain.

## Codex ECC Adaptation

- [x] Inspect the current project-level Codex guidance and note Claude-only behaviors that should not survive the migration.
- [x] Review the upstream `everything-claude-code` Codex support files and extract the parts that translate cleanly to this repo.
- [ ] Rewrite the local instruction stack for Codex-native planning, verification, review, and project memory.
- [ ] Add project-local `.codex` baseline files tuned to Helion kernel work instead of generic web-app defaults.
- [ ] Add the missing `tasks/lessons.md` scaffold referenced by the project rules.
- [ ] Review the final guidance for contradictions with current project constraints and document the result below.

## Codex ECC Adaptation Review

- Pending implementation.
