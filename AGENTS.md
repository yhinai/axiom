# Project Rules for Codex

## Auto-Commit and Push Rule

**MANDATORY**: After every change you make to any file in this repository, you MUST:

1. Stage the changed files: `git add <specific files you changed>`
2. Commit with a clear message describing what changed: `git commit -m "description of change"`
3. Push to `main`: `git push origin main`

This applies to EVERY change. Commit and push immediately after each logical change.

- Always push to `main`
- Never force push
- Use descriptive commit messages that explain the why
- If a pre-commit hook fails, fix the issue and create a new commit rather than amending

## Codex Operating Model

- Plan before non-trivial work. Use a short explicit plan for any task with 3 or more steps, architectural risk, or broad file impact.
- Mirror that plan in `tasks/todo.md` with checkable items before implementation starts.
- Keep work moving locally unless parallel Codex agents are clearly helpful and ownership can stay disjoint.
- After implementation, do a review pass focused on correctness, performance regressions, shape coverage, and missing verification.
- Capture durable project knowledge in the existing docs structure instead of creating new top-level notes by default.

## Parallel Work Strategy

Use Codex subagents, separate worktrees, or parallel terminals only when the task has independent modules and clear ownership boundaries.

### Good Fits for Parallel Work

- Multi-kernel optimization in parallel, with one worker per kernel
- Research plus implementation, with one worker gathering evidence while another edits
- Debugging with competing hypotheses on different kernels or scripts
- Shared infra work split away from kernel-local tuning

### Bad Fits for Parallel Work

- Sequential tasks with heavy dependencies between steps
- Changes isolated to a single `submission.py`
- Simple config tweaks or tiny fixes
- Tasks where coordination overhead exceeds the likely speedup

### Recommended Team Shape

- Start with 2-4 workers plus a lead for most larger efforts
- Give each worker a disjoint file or module boundary
- Prefer read-only explorers and reviewers unless a worker owns a bounded write scope
- If the current Codex session or policy does not allow delegation, keep the same ownership boundaries in the plan and execute sequentially

### Independent Modules

| Module | Directory | Notes |
|--------|-----------|-------|
| Causal Conv1D | `causal_conv1d_py/` | Small inner loop, pointer indexing |
| Chunk Fwd H | `gated_deltanet_chunk_fwd_h_py/` | Dot-heavy, tensor_descriptor, num_ctas=2 |
| Chunk Fwd O | `gated_deltanet_chunk_fwd_o_py/` | Dot-heavy, tensor_descriptor, num_ctas=2 |
| Recompute W/U | `gated_deltanet_recompute_w_u_py/` | Dot-heavy, tensor_descriptor, num_ctas=2 |
| Leaderboard TUI | `leaderboard-tui/` | Go app for watching rankings |
| Shared Infra | `eval.py`, `utils.py` | Test and benchmark framework |

### Coordination Rules

- Record active work, ownership, and blockers in `tasks/todo.md`
- Express dependencies in the plan so blocked work is obvious
- Finished work is not done until verification passes
- Handoffs should include touched files, commands run, and any remaining risk
- Before shutdown or handoff, ensure changes are committed and pushed

## Workflow Orchestration

### 1. Plan First

- For non-trivial tasks, write the plan to `tasks/todo.md` before editing code
- If assumptions change or verification fails, stop and re-plan instead of pushing through
- Use the plan for verification work too, not just implementation

### 2. Reproducer or Test First When Practical

- Prefer failing tests, minimal repros, or benchmark deltas before changing behavior
- For kernel tuning, the repro can be an `eval.py` correctness failure, a benchmark regression, or a missing-shape dispatch gap
- For docs or workflow changes, define the inconsistency being fixed before editing

### 3. Review After Writing

- Review every meaningful change for correctness, regressions, security, and missing verification
- Style-only cleanup is lower priority than dispatch bugs, wrong shapes, invalid configs, or benchmark regressions
- When a change is risky, compare behavior against the current `main` branch expectation

### 4. Verification Before Done

- Never mark a task complete without proving it works
- Diff behavior between the old and new path when relevant
- Ask whether the change would survive staff-level review for correctness and maintainability

### 5. Self-Improvement Loop

- After any user correction or preventable process mistake, update `tasks/lessons.md`
- Write short rules that would have prevented the mistake
- Review relevant lessons at the start of related future work

### 6. Demand Elegance

- Prefer the simplest fix that also generalizes cleanly
- If a solution feels hacky, pause and look for the cleaner structural change
- Do not over-engineer obvious fixes

### 7. Autonomous Bug Fixing

- When given a bug report, investigate the actual failure and fix it end to end
- Point at the reproducer, failing command, or benchmark evidence, then resolve it
- Avoid making the user orchestrate the debugging session

## Task Management

1. **Plan First**: Write the active plan to `tasks/todo.md` with checkable items.
2. **Track Progress**: Mark items complete as you go.
3. **Document Results**: Add a review section to `tasks/todo.md` for what changed, what passed, and what was rejected.
4. **Capture Lessons**: Update `tasks/lessons.md` after user corrections or process misses.
5. **Prefer Existing Docs**: Extend the current docs set before adding new top-level markdown files.

## Project Context

- **Competition**: GPU MODE Helion Hackathon on NVIDIA B200 GPUs
- **Language**: Python, with Helion DSL compiling to Triton or TileIR
- **Kernels**: `causal_conv1d`, `gated_deltanet_chunk_fwd_h`, `gated_deltanet_chunk_fwd_o`, `gated_deltanet_recompute_w_u`
- **Framework**: `helion.kernel`, `helion.language as hl`
- **Test and Bench**: `python eval.py test|benchmark|both <kernel_dir>/`
- **Submission**: `popcorn submit submission.py --gpu B200_Nebius --leaderboard <name>`
- **Remote Server**: use the existing repo scripts and local environment; do not paste credentials into new docs or logs
- **Leaderboard TUI**: `./leaderboard-tui/leaderboard-tui`

### Submission Pattern

Every `submission.py` should follow this structure:

```python
#!POPCORN leaderboard <kernel_name>
#!POPCORN gpu B200_Nebius

SHAPE_CONFIGS: dict[tuple, helion.Config] = { ... }
_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}
def custom_kernel(data: input_t) -> output_t: ...
```

### Config Tuning

- `block_sizes` control tile dimensions
- `num_warps` is 4 for TileIR and 1-32 otherwise
- `num_stages` controls pipeline depth
- `indexing` should be `"pointer"` for simple elementwise kernels and `"tensor_descriptor"` for dot-heavy kernels
- `pid_type` can be `"flat"`, `"linear"`, or `"persistent_blocked"`
- TileIR-only knobs include `num_ctas` and `occupancy`

### Competition Rules

- All configs must be hardcoded; no runtime autotuning on KernelBot
- Helion DSL should remain at least 70 percent of kernel code
- Accuracy targets are `1e-2` for deltanet, `1e-3` for fp8, and `1e-5` for conv1d
- Scoring is 100 points for correctness plus up to 1000 for performance

## Verification Standards

Before marking a kernel task complete:

- Run `python eval.py test <kernel_dir>/`
- Run `python eval.py benchmark <kernel_dir>/`
- Verify `SHAPE_CONFIGS` covers all test and benchmark shapes from `task.yml`
- Check that `custom_kernel` dispatches to the correct kernel for each supported shape
- Confirm the change does not violate the hardcoded-config competition rule

For shared infra changes:

- Run focused verification for the affected scripts or helpers
- Re-run the impacted kernel evaluations rather than assuming harness-level changes are safe

## Security and Git Hygiene

- Never introduce new hardcoded secrets or reprint existing credentials into docs, logs, or summaries
- Review `git diff` before committing and pushing
- Never force push or rewrite user commits
- If the remote, branch, or destination looks unexpected, pause and confirm before pushing

## Core Principles

- **Simplicity First**: keep changes as small and direct as possible
- **No Laziness**: find root causes rather than layering temporary fixes
- **Minimal Impact**: touch only what is necessary and avoid collateral regressions
- **Evidence Over Guesswork**: prefer measured benchmark data, failing repros, and exact shape coverage over intuition
- **Competition Realism**: optimize for verified leaderboard-safe wins, not speculative micro-optimizations
