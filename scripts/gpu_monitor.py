#!/usr/bin/env python3
"""Write one-second NVIDIA GPU utilization samples as JSONL."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    return parser.parse_args()


def sample() -> dict:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    gpu, mem, used, total = [int(part.strip()) for part in output.split(",")]
    return {
        "time": time.time(),
        "gpu_util": gpu,
        "mem_util": mem,
        "mem_used_mib": used,
        "mem_total_mib": total,
    }


def main() -> int:
    args = parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        with path.open("a") as handle:
            handle.write(json.dumps(sample(), sort_keys=True) + "\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
