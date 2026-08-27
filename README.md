<div align="center">

# Axiom

### Fast kernels. Measured on real silicon.

Helion kernels for Mamba-2 and Gated DeltaNet, shaped for NVIDIA B200.

[Results](#results) · [Quick start](#quick-start) · [Kernels](#the-kernels) · [Deep dives](#deep-dives)

</div>

<br>

<p align="center">
  <img src="assets/b200_gpu_utilization.png" alt="NVIDIA B200 utilization during Axiom optimizer trials" width="920">
</p>

<p align="center"><sub>Live B200 optimizer trials: compile, verify, benchmark, repeat.</sub></p>

<br>

## A smaller path to faster sequence models

Axiom turns four expensive sequence operations into specialized GPU kernels.
Each implementation is written in the [Helion](https://github.com/pytorch/helion)
DSL, compiled for the target shape, checked against a PyTorch reference, and
timed with the same evaluation harness.

```text
inputs
  └─ recompute W/U ──► chunk state ──► chunk output
         WY transform     recurrence      attention + state

     causal Conv1D ──► independent Mamba-style sequence path
```

## Results

Measured on the `helion` H200 environment with `python eval.py benchmark`.
The baseline and optimized kernels use the same harness and benchmark shapes.

| Kernel | Baseline | Axiom | Geomean speedup |
|:--|--:|--:|--:|
| Causal Conv1D | 249.5 µs | 9.5 µs | **29.0×** |
| Recompute W/U | 359.0 µs | 5.6 µs | **57.8×** |

<sub>Values above are the fastest minimum latency shown for each kernel; the
speedup column is the geometric mean across all three benchmark shapes. See the
[full comparison](PRESENTATION_BASELINE_VS_OPTIMIZED.md) for per-shape results
and methodology.</sub>

### What changed

| Principle | Implementation |
|:--|:--|
| Do less work | Removed host padding, duplicate accumulation, and redundant loads |
| Match the math | Recast WY transforms as tensor-core matrix multiplies |
| Specialize early | Hardcoded shape-aware tiles, warps, stages, and indexing |
| Keep data close | Fused accumulation, persistent blocks, and cache-aware grouping |

## Quick start

Run on a CUDA machine with Python, PyTorch, and Helion installed.

```bash
git clone https://github.com/yhinai/axiom.git
cd axiom

# Correctness + benchmark
python eval.py both causal_conv1d_py/
python eval.py both gated_deltanet_recompute_w_u_py/
```

Run the full four-kernel suite:

```bash
python run_all_kernels.py
```

## The kernels

### 01 · Causal Conv1D

Depthwise convolution with causal masking performed inside the kernel—no
materialized padding, duplicate accumulation, or unnecessary memory traffic.

[`causal_conv1d_py/submission.py`](causal_conv1d_py/submission.py)

### 02 · Recompute W/U

The Gated DeltaNet WY transform expressed as two direct `hl.dot` operations,
mapping the dominant work to tensor cores.

[`gated_deltanet_recompute_w_u_py/submission.py`](gated_deltanet_recompute_w_u_py/submission.py)

### 03 · Chunk Fwd H

The inter-chunk state recurrence. Gate scaling, state decay, correction, and
matrix accumulation stay inside one shape-specialized kernel.

[`gated_deltanet_chunk_fwd_h_py/submission.py`](gated_deltanet_chunk_fwd_h_py/submission.py)

### 04 · Chunk Fwd O

The final output stage, combining causal intra-chunk attention with the global
state contribution in a single pass.

[`gated_deltanet_chunk_fwd_o_py/submission.py`](gated_deltanet_chunk_fwd_o_py/submission.py)

## Deep dives

- [Optimization ladder](PRESENTATION_OPTIMIZATION_LADDER.md) — each cumulative change and its measured effect.
- [Baseline vs. optimized](PRESENTATION_BASELINE_VS_OPTIMIZED.md) — methodology and per-shape timings.
- [Complete Helion report](HELION_COMPLETE_REPORT.md) — implementation details and broader findings.
- [External kernel review](EXTERNAL_KERNEL_REVIEW.md) — adopted, rejected, and benchmarked ideas.

## Optimizer

Axiom includes a verifier-first overnight loop for B200. A candidate is accepted
only when it passes correctness and improves geometric mean latency. Every trial
is stored as JSONL and can be streamed to HUD.

```bash
python scripts/run_modal_overnight_hud.py \
  --duration-hours 8 \
  --workers 8 \
  --hud-concurrency 8 \
  --job-name axiom-b200-overnight
```

<details>
<summary><strong>Optimizer artifacts</strong></summary>

| Path | Contents |
|:--|:--|
| `runs/modal-b200-axiom/*/trials.jsonl` | Every verified candidate |
| `runs/modal-b200-axiom/*/best_history.jsonl` | Accepted improvements |
| `runs/modal-b200-axiom/*/best_submission.py` | Best accepted kernel |
| `runs/modal-b200-axiom/gpu_utilization.jsonl` | One-second GPU samples |
| `runs/logs/` | Optimizer and HUD stream logs |

</details>

## Repository map

```text
axiom/
├── causal_conv1d_py/                  # Mamba-style depthwise convolution
├── gated_deltanet_recompute_w_u_py/  # WY transform
├── gated_deltanet_chunk_fwd_h_py/    # recurrent state
├── gated_deltanet_chunk_fwd_o_py/    # final output
├── scripts/                           # optimizer, telemetry, HUD stream
├── leaderboard-tui/                   # live leaderboard viewer
└── eval.py                            # shared correctness + benchmark harness
```

## Acknowledgments

Built for the GPU MODE Helion Hackathon. Baselines are derived from
[`gpu-mode/reference-kernels`](https://github.com/gpu-mode/reference-kernels/tree/main/problems/helion).

<div align="center">

<sub>Built to be measured.</sub>

</div>
