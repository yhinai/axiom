# Codex Supplement for Helion

This file supplements the root `AGENTS.md` with Codex-native guidance adapted from the useful parts of Everything Claude Code.

## ECC Concepts Mapped to Codex

- **Planner** maps to `update_plan` plus a mirrored checklist in `tasks/todo.md`
- **TDD guide** maps to reproducer-first work: write a failing test, isolate a benchmark regression, or prove a missing-shape dispatch before changing code
- **Code reviewer** maps to an explicit self-review pass focused on correctness, regressions, performance, and missing verification
- **Security reviewer** maps to diff review, secret hygiene, and push safety before every commit

## Recommended Codex Roles

- `explorer`: read-only evidence gathering across `submission.py`, `task.yml`, `eval.py`, and benchmarking helpers
- `reviewer`: read-only review for correctness, missing shapes, config regressions, and secret exposure
- `benchmarker`: read-only performance specialist for `eval.py`, leaderboard readiness, and benchmark delta analysis

## Helion Verification Checklist

- Match all shapes listed in each kernel's `task.yml`
- Keep configs hardcoded inside `submission.py`
- Preserve the correct tolerance target for the kernel family
- Validate changed kernels with `python eval.py test|benchmark|both <kernel_dir>/`
- When evaluating external ideas, record both adopted wins and rejected variants in the existing docs or `tasks/todo.md`

## Research and Web Use

- Verify changing APIs, leaderboard behavior, and external optimization claims against primary sources before relying on them
- Prefer upstream code, official docs, and the exact submission or PR being referenced
- Use current web lookups for time-sensitive facts such as competition rules, leaderboard endpoints, or tool behavior

## Without Hooks

- Review `git diff` manually before each commit and push
- Do not assume tests, security scans, or secret checks are enforced automatically
- Use project-local `.codex/config.toml` as the baseline and keep user-level config reserved for personal extras
