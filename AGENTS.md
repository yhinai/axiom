# Project Rules for Codex

## Auto-Commit and Push Rule

**MANDATORY**: After every change you make to any file in this repository, you MUST:

1. Stage the changed files: `git add <specific files you changed>`
2. Commit with a clear message describing what changed: `git commit -m "description of change"`
3. Push to `main`: `git push origin main`

This applies to EVERY change — no exceptions. Do not batch changes. Commit and push immediately after each logical change.

- Always push to `main`
- Never force push
- Use descriptive commit messages that explain the "why"
- If a pre-commit hook fails, fix the issue and create a NEW commit (never amend)

## Agent Team Strategy

Use agent teams for any task that benefits from parallel work across independent modules. Teams are enabled via `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` in settings.

### When to Use Teams
- Multi-kernel optimization in parallel (one teammate per kernel)
- Research + implementation in parallel (one teammate explores TileIR, another tunes configs)
- Debugging with competing hypotheses — teammates test different theories simultaneously
- Any task with 3+ independent subtasks that don't touch the same files

### When NOT to Use Teams
- Sequential tasks with heavy dependencies between steps
- Changes to a single kernel's submission.py
- Simple config tweaks or small fixes
- Tasks where coordination overhead exceeds the benefit

### Team Configuration
- Start with **3-5 teammates** for most workflows
- Aim for **5-6 tasks per teammate** to keep everyone productive
- Use **Opus for the lead** (reasoning/coordination), **Opus for teammates** (focused implementation)
- Use **delegate mode** (`Shift+Tab`) when the lead should only coordinate, not write code

### Independent Modules

Each teammate should own a separate kernel to avoid file conflicts:

| Module | Directory | Notes |
|--------|-----------|-------|
| Causal Conv1D | `causal_conv1d_py/` | Small inner loop, pointer indexing |
| Chunk Fwd H | `gated_deltanet_chunk_fwd_h_py/` | Dot-heavy, tensor_descriptor, num_ctas=2 |
| Chunk Fwd O | `gated_deltanet_chunk_fwd_o_py/` | Dot-heavy, tensor_descriptor, num_ctas=2 |
| Recompute W/U | `gated_deltanet_recompute_w_u_py/` | Dot-heavy, tensor_descriptor, num_ctas=2 |
| Leaderboard TUI | `leaderboard-tui/` | Go app for watching rankings |
| Shared Infra | `eval.py`, `utils.py` | Test/benchmark framework |

### Team Communication Rules
- Use `SendMessage` (type: "message") for direct teammate communication — always refer to teammates by **name**
- Use `SendMessage` (type: "broadcast") **only** for critical blockers affecting everyone
- Use `TaskCreate`/`TaskUpdate`/`TaskList` for work coordination — teammates self-claim unblocked tasks
- When a teammate finishes, they check `TaskList` for the next available task (prefer lowest ID first)
- Mark tasks `completed` only after verification passes

### Task Dependencies
- Use `addBlockedBy` to express task ordering (e.g., "chunk_fwd_o depends on chunk_fwd_h being correct")
- Teammates skip blocked tasks and pick up unblocked work
- When a blocking task completes, dependent tasks auto-unblock

### Plan Approval for Risky Work
- For architectural changes or risky refactors, require **plan approval** before implementation
- The teammate works in read-only mode, submits a plan, lead approves/rejects
- Only after approval does the teammate implement

### Team Quality Hooks
- `TaskCompleted` hook: prevents marking tasks done unless tests pass
- `TeammateIdle` hook: auto-assigns follow-up work to idle teammates
- Every teammate must run verification before reporting completion

### Shutdown Protocol
- When all tasks are complete, the lead sends `shutdown_request` to each teammate
- Teammates approve shutdown after confirming their work is committed
- Lead calls `TeamDelete` to clean up team resources

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Project Context

- **Competition**: GPU MODE Helion Hackathon on NVIDIA B200 (Blackwell) GPUs
- **Language**: Python (Helion DSL compiling to Triton/TileIR)
- **Kernels**: 4 GPU kernels (causal_conv1d, 3x gated_deltanet)
- **Framework**: Helion DSL (`helion.kernel`, `helion.language as hl`)
- **Test/Bench**: `python eval.py test|benchmark|both <kernel_dir>/`
- **Submit**: `popcorn submit submission.py --gpu B200_Nebius --leaderboard <name>`
- **Remote Server**: `sshpass -p 'bMvoEtw1B6' ssh ubuntu@46.243.147.105`
- **Leaderboard TUI**: `./leaderboard-tui/leaderboard-tui` (Go/Bubble Tea)

### Submission Pattern

Every `submission.py` follows this structure:
```python
#!POPCORN leaderboard <kernel_name>
#!POPCORN gpu B200_Nebius

SHAPE_CONFIGS: dict[tuple, helion.Config] = { ... }
_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}
def custom_kernel(data: input_t) -> output_t: ...
```

### Config Tuning

- `block_sizes` — tile dimensions
- `num_warps` — 4 (fixed for TileIR), 1-32 otherwise
- `num_stages` — pipeline depth (1-7)
- `indexing` — `"pointer"` (elementwise) or `"tensor_descriptor"` (dot-heavy)
- `pid_type` — `"flat"`, `"linear"`, or `"persistent_blocked"`
- TileIR: `ENABLE_TILE=1 HELION_BACKEND=tileir`, `num_ctas` (1-2), `occupancy` (1-8)

### Competition Rules

- All configs must be **hardcoded** (no runtime autotuning on KernelBot)
- Helion DSL >= 70% of kernel code
- Accuracy: rtol/atol = 1e-2 (deltanet), 1e-3 (fp8), 1e-5 (conv1d)
- Scoring: 100pts correctness + 0-1000pts performance (top 10 only)

## Verification Standards

Before marking any kernel task complete:
- `python eval.py test <kernel_dir>/` — all test shapes pass correctness
- `python eval.py benchmark <kernel_dir>/` — performance numbers look reasonable
- Verify `SHAPE_CONFIGS` covers all test AND benchmark shapes from `task.yml`
- Check that `custom_kernel` dispatches to the correct kernel for each shape

## Leaderboard API

- Base: `https://site--bot--dxfjds728w5v.code.run`
- Rankings: `GET /submissions/<kernel_name>/B200_Nebius` (header: `X-Popcorn-Cli-Id`)
- Your submissions: `GET /user/submissions`

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
