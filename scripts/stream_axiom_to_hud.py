#!/usr/bin/env python3
"""Stream Axiom optimizer JSONL trial rows into one HUD job."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_hud_key() -> None:
    if os.environ.get("HUD_API_KEY"):
        return
    path = Path.home() / ".hud" / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "HUD_API_KEY":
            os.environ["HUD_API_KEY"] = value.strip().strip('"').strip("'")
            return


def _message(role: str, text: str):
    import mcp.types as mcp_types

    return mcp_types.PromptMessage(
        role=role,
        content=mcp_types.TextContent(type="text", text=text),
    )


class TrialRowAgent:
    def __init__(self, row: dict[str, Any]):
        self.row = row

    async def __call__(self, run) -> None:
        from hud.types import Step

        payload = json.dumps(self.row, sort_keys=True)
        run.trace.content = payload
        run.trace.extra.update(
            {
                "agent": "axiom_jsonl_streamer",
                "kernel": self.row.get("kernel"),
                "trial": self.row.get("trial"),
                "candidate": self.row.get("candidate"),
                "correct": self.row.get("correct"),
                "accepted": self.row.get("accepted"),
                "reward": self.row.get("reward"),
                "candidate_geomean_mean_ms": self.row.get("candidate_geomean_mean_ms"),
                "best_geomean_mean_ms_after": self.row.get("best_geomean_mean_ms_after"),
            }
        )
        run.record(
            Step(
                source="agent",
                messages=[_message("assistant", payload)],
                extra={"event": "axiom_trial_row", **run.trace.extra},
            )
        )


def _load_state(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text()))
    except Exception:
        return set()


def _save_state(path: Path, sent: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(sent), indent=2) + "\n")


def _row_key(path: Path, line_no: int, row: dict[str, Any]) -> str:
    return f"{path}:{line_no}:{row.get('kernel')}:{row.get('trial')}:{row.get('candidate')}"


def _iter_new_rows(run_dir: Path, sent: set[str]):
    for path in sorted(run_dir.glob("*/improvements_*.jsonl")):
        for idx, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "trial":
                continue
            key = _row_key(path, idx, row)
            if key in sent:
                continue
            yield key, row


async def _run(args: argparse.Namespace) -> int:
    _bootstrap_hud_key()
    if not os.environ.get("HUD_API_KEY"):
        print("HUD_API_KEY is required", file=sys.stderr)
        return 2

    from hud.eval import Job, LocalRuntime, Taskset

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    state_path = Path(args.state)
    if not state_path.is_absolute():
        state_path = run_dir / args.state
    sent = _load_state(state_path)

    taskset = Taskset.from_file(args.source).filter(["axiom_optimizer_trial"])
    job = await Job.start(args.job_name, group=1)
    print(f"https://hud.ai/jobs/{job.id}", flush=True)

    started = time.time()
    streamed = 0

    async def stream_one(row: dict[str, Any]) -> str:
        hud_job = await taskset.run(
            TrialRowAgent(row),
            runtime=LocalRuntime(args.source),
            job=job,
            group=1,
            max_concurrent=1,
            rollout_timeout=args.rollout_timeout,
        )
        return hud_job.id

    while True:
        new_rows = list(_iter_new_rows(run_dir, sent))
        did_work = bool(new_rows)
        for offset in range(0, len(new_rows), args.max_concurrent):
            batch = new_rows[offset : offset + args.max_concurrent]
            results = await asyncio.gather(
                *(stream_one(row) for _, row in batch),
                return_exceptions=True,
            )
            for (key, row), result in zip(batch, results, strict=True):
                if isinstance(result, Exception):
                    print(
                        json.dumps(
                            {
                                "stream_error": {"type": type(result).__name__, "message": str(result)},
                                "kernel": row.get("kernel"),
                                "trial": row.get("trial"),
                                "candidate": row.get("candidate"),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
                sent.add(key)
                streamed += 1
                _save_state(state_path, sent)
                print(
                    json.dumps(
                        {
                            "streamed": streamed,
                            "job_id": result,
                            "kernel": row.get("kernel"),
                            "trial": row.get("trial"),
                            "candidate": row.get("candidate"),
                            "accepted": row.get("accepted"),
                            "reward": row.get("reward"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if args.max_rows and streamed >= args.max_rows:
                    return 0
        if args.once and not did_work:
            return 0
        if args.duration_seconds and time.time() - started >= args.duration_seconds:
            return 0
        await asyncio.sleep(args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/modal-b200-axiom")
    parser.add_argument("--source", default="hud_env.py")
    parser.add_argument("--job-name", default="axiom-modal-b200-live")
    parser.add_argument("--state", default=".hud_stream_state.json")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--rollout-timeout", type=float, default=120.0)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    return asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
