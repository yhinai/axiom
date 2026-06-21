#!/usr/bin/env python3
"""Start Axiom's Modal B200 optimizer and HUD live stream."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/workspace/protean/.venv/bin/python"


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def ssh(remote: str, command: str) -> None:
    run(["ssh", remote, command])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default="modal")
    parser.add_argument("--remote-dir", default="/workspace/axiom")
    parser.add_argument("--duration-hours", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hud-concurrency", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON)
    parser.add_argument("--job-name", default="axiom-b200-overnight")
    parser.add_argument("--optimizer-session", default="axiom-kernel-improve")
    parser.add_argument("--hud-session", default="axiom-hud-stream")
    parser.add_argument("--gpu-session", default="axiom-gpu-monitor")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"runs/modal-b200-axiom-{timestamp}"

    env = os.environ.copy()
    env.update(
        {
            "REMOTE_HOST": args.remote,
            "REMOTE_DIR": args.remote_dir,
            "SESSION": args.optimizer_session,
            "DURATION_HOURS": str(args.duration_hours),
            "OUT_DIR": out_dir,
            "PYTHON_BIN": args.python_bin,
            "KERNEL_WORKERS": str(args.workers),
        }
    )
    run(["bash", "scripts/start_modal_axiom_optimizer.sh"], env=env)

    quoted_remote_dir = shlex.quote(args.remote_dir)
    quoted_out_dir = shlex.quote(out_dir)
    quoted_job = shlex.quote(args.job_name)
    python_bin = shlex.quote(args.python_bin)

    ssh(
        args.remote,
        (
            f"cd {quoted_remote_dir} && mkdir -p runs/logs && "
            f"tmux kill-session -t {shlex.quote(args.gpu_session)} 2>/dev/null || true; "
            f"tmux new-session -d -s {shlex.quote(args.gpu_session)} "
            f"'cd {quoted_remote_dir} && {python_bin} scripts/gpu_monitor.py "
            f"--out {quoted_out_dir}/gpu_utilization.jsonl "
            f"2>&1 | tee runs/logs/{shlex.quote(args.gpu_session)}.log'"
        ),
    )

    ssh(
        args.remote,
        (
            f"cd {quoted_remote_dir} && mkdir -p runs/logs && "
            f"tmux kill-session -t {shlex.quote(args.hud_session)} 2>/dev/null || true; "
            f"tmux new-session -d -s {shlex.quote(args.hud_session)} "
            f"\"cd {quoted_remote_dir} && {python_bin} scripts/stream_axiom_to_hud.py "
            f"--run-dir {quoted_out_dir} --source hud_app/env.py --job-name {quoted_job} "
            f"--state .hud_stream_state.json --poll-seconds {args.poll_seconds} "
            f"--max-concurrent {args.hud_concurrency} "
            f"2>&1 | tee runs/logs/{shlex.quote(args.hud_session)}.log\""
        ),
    )

    print(
        "\n".join(
            [
                "Started Axiom Modal overnight HUD run.",
                f"  remote: {args.remote}",
                f"  out:    {args.remote_dir}/{out_dir}",
                f"  job:    {args.job_name}",
                "",
                "Watch:",
                f"  ssh {args.remote} 'tail -f {args.remote_dir}/runs/logs/{args.optimizer_session}.log'",
                f"  ssh {args.remote} 'tail -f {args.remote_dir}/runs/logs/{args.hud_session}.log'",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
