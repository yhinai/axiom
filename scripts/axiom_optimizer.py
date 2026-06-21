#!/usr/bin/env python3
"""Iteratively improve Axiom Helion kernels with verifier-first logging.

This runner treats each candidate ``submission.py`` as a deployable kernel:
copy it into the target kernel directory, run Axiom's real ``eval.py both``
harness, parse correctness and benchmark timings, then keep the candidate only
when it improves geometric mean latency.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERNELS = (
    "causal_conv1d_py",
    "gated_deltanet_chunk_fwd_h_py",
    "gated_deltanet_chunk_fwd_o_py",
    "gated_deltanet_recompute_w_u_py",
)
BENCHMARK_RE = re.compile(
    r"^\s*Benchmark\s+\d+:\s+"
    r"(?P<mean>[0-9.]+)\s+ms\s+"
    r"\(min=(?P<min>[0-9.]+),\s+max=(?P<max>[0-9.]+)\)\s+"
    r"(?P<spec>.+)$"
)
TEST_RE = re.compile(r"^\s*Test\s+\d+:\s+(?P<status>PASS|FAIL)\b")


@dataclass(frozen=True)
class Candidate:
    kernel: str
    name: str
    source_path: Path
    origin: str


def geometric_mean(values: list[float]) -> float:
    if not values:
        return math.inf
    if any(value <= 0 for value in values):
        return math.inf
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_eval_output(output: str, return_code: int) -> dict:
    tests_passed = True
    benchmark_failures = 0
    benchmarks = []
    for line in output.splitlines():
        test_match = TEST_RE.match(line)
        if test_match and test_match.group("status") == "FAIL":
            tests_passed = False
        if "Some tests FAILED." in line or "FAIL (correctness)" in line:
            tests_passed = False
        if "FAIL (correctness)" in line:
            benchmark_failures += 1
        benchmark_match = BENCHMARK_RE.match(line)
        if benchmark_match:
            benchmarks.append(
                {
                    "mean_ms": float(benchmark_match.group("mean")),
                    "min_ms": float(benchmark_match.group("min")),
                    "max_ms": float(benchmark_match.group("max")),
                    "spec": benchmark_match.group("spec"),
                }
            )
    correct = return_code == 0 and tests_passed and benchmark_failures == 0 and bool(benchmarks)
    geomean_mean_ms = geometric_mean([row["mean_ms"] for row in benchmarks])
    geomean_min_ms = geometric_mean([row["min_ms"] for row in benchmarks])
    return {
        "return_code": return_code,
        "correct": correct,
        "tests_passed": tests_passed,
        "benchmark_failures": benchmark_failures,
        "benchmark_count": len(benchmarks),
        "geomean_mean_ms": None if math.isinf(geomean_mean_ms) else round(geomean_mean_ms, 9),
        "geomean_min_ms": None if math.isinf(geomean_min_ms) else round(geomean_min_ms, 9),
        "benchmarks": benchmarks,
    }


def discover_candidates(kernel: str) -> list[Candidate]:
    candidates = [Candidate(kernel, "current", ROOT / kernel / "submission.py", "current")]
    search_roots = [ROOT / "external_submissions", ROOT / "experiments"]
    for root in search_roots:
        if not root.exists():
            continue
        for source in sorted(root.glob(f"*/{kernel}/submission.py")):
            name = f"{source.parents[1].name}_{source.parents[0].name}"
            candidates.append(Candidate(kernel, name, source, str(source.relative_to(ROOT))))
    seen = set()
    unique = []
    for candidate in candidates:
        key = candidate.source_path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_candidate(
    *,
    python_exe: str,
    kernel: str,
    candidate: Candidate,
    timeout: float,
    out_dir: Path,
    trial: int,
) -> dict:
    kernel_dir = ROOT / kernel
    submission_path = kernel_dir / "submission.py"
    deployed_dir = out_dir / kernel / "deployed"
    deployed_dir.mkdir(parents=True, exist_ok=True)
    deployed_path = deployed_dir / f"{trial:04d}_{candidate.name}.py"
    shutil.copy2(candidate.source_path, deployed_path)
    if candidate.source_path.resolve() != submission_path.resolve():
        shutil.copy2(candidate.source_path, submission_path)

    started = time.time()
    process = subprocess.run(
        [python_exe, "eval.py", "both", f"{kernel}/"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    parsed = parse_eval_output(process.stdout, process.returncode)
    output_path = out_dir / kernel / "outputs" / f"{trial:04d}_{candidate.name}.log"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(process.stdout)
    return {
        "candidate": candidate.name,
        "origin": candidate.origin,
        "source_path": str(candidate.source_path.relative_to(ROOT)),
        "deployed_path": str(deployed_path.relative_to(ROOT)),
        "output_path": str(output_path.relative_to(ROOT)),
        "elapsed_sec": round(time.time() - started, 6),
        **parsed,
    }


def optimize_kernel(args: argparse.Namespace, kernel: str) -> dict:
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    kernel_out = out_dir / kernel
    kernel_out.mkdir(parents=True, exist_ok=True)
    log_path = kernel_out / "trials.jsonl"
    improvements_path = kernel_out / f"improvements_{kernel}.jsonl"
    best_path = kernel_out / "best_submission.py"

    submission_path = ROOT / kernel / "submission.py"
    original_source = submission_path.read_text()
    backup_path = kernel_out / "original_submission.py"
    backup_path.write_text(original_source)

    candidates = discover_candidates(kernel)
    if args.max_candidates:
        candidates = candidates[: args.max_candidates]

    deadline = time.time() + args.duration_hours * 3600 if args.duration_hours else None
    best_score_ms = math.inf
    best_candidate = None
    trial = 0
    accepted = 0
    started = time.time()

    try:
        while True:
            for candidate in candidates:
                if deadline is not None and time.time() >= deadline:
                    raise TimeoutError
                trial += 1
                before_ms = best_score_ms
                try:
                    result = run_candidate(
                        python_exe=args.python,
                        kernel=kernel,
                        candidate=candidate,
                        timeout=args.timeout,
                        out_dir=out_dir,
                        trial=trial,
                    )
                    eval_error = None
                except subprocess.TimeoutExpired as exc:
                    result = {
                        "candidate": candidate.name,
                        "origin": candidate.origin,
                        "source_path": str(candidate.source_path.relative_to(ROOT)),
                        "elapsed_sec": args.timeout,
                        "return_code": 124,
                        "correct": False,
                        "tests_passed": False,
                        "benchmark_failures": 1,
                        "benchmark_count": 0,
                        "geomean_mean_ms": None,
                        "geomean_min_ms": None,
                        "benchmarks": [],
                    }
                    eval_error = {"type": "TimeoutExpired", "message": str(exc)}

                candidate_ms = result.get("geomean_mean_ms")
                is_better = bool(result["correct"] and candidate_ms is not None and candidate_ms < best_score_ms)
                best_candidate_before = best_candidate
                if is_better:
                    accepted += 1
                    best_score_ms = float(candidate_ms)
                    best_candidate = candidate.name
                    shutil.copy2(ROOT / kernel / "submission.py", best_path)
                elif best_path.exists():
                    shutil.copy2(best_path, submission_path)
                else:
                    submission_path.write_text(original_source)

                speedup_vs_best = None
                if candidate_ms and not math.isinf(before_ms):
                    speedup_vs_best = round(before_ms / float(candidate_ms), 6)
                row = {
                    "event": "trial",
                    "time": time.time(),
                    "kernel": kernel,
                    "trial": trial,
                    "accepted": is_better,
                    "best_candidate_before": best_candidate_before,
                    "best_geomean_mean_ms_before": None if math.isinf(before_ms) else round(before_ms, 9),
                    "best_candidate_after": best_candidate,
                    "best_geomean_mean_ms_after": None if math.isinf(best_score_ms) else round(best_score_ms, 9),
                    "speedup_vs_best_before": speedup_vs_best,
                    "reward": 0.0 if not result["correct"] else (speedup_vs_best or 1.0),
                    "eval_error": eval_error,
                    **result,
                }
                write_jsonl(log_path, row)
                write_jsonl(improvements_path, {
                    "event": "trial",
                    "time": row["time"],
                    "kernel": kernel,
                    "trial": trial,
                    "candidate": candidate.name,
                    "accepted": is_better,
                    "correct": result["correct"],
                    "reward": row["reward"],
                    "candidate_geomean_mean_ms": candidate_ms,
                    "best_geomean_mean_ms_after": row["best_geomean_mean_ms_after"],
                    "speedup_vs_best_before": speedup_vs_best,
                })
                if args.once:
                    raise TimeoutError
            if not args.loop:
                break
    except TimeoutError:
        pass
    finally:
        if best_path.exists():
            shutil.copy2(best_path, submission_path)
        else:
            submission_path.write_text(original_source)

    final = {
        "event": "run_complete",
        "time": time.time(),
        "kernel": kernel,
        "trials": trial,
        "accepted": accepted,
        "best_candidate": best_candidate,
        "best_geomean_mean_ms": None if math.isinf(best_score_ms) else round(best_score_ms, 9),
        "elapsed_sec": round(time.time() - started, 6),
        "log": str(log_path),
        "improvements": str(improvements_path),
    }
    write_jsonl(log_path, final)
    write_jsonl(improvements_path, final)
    (kernel_out / "summary.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernels", default=",".join(DEFAULT_KERNELS))
    parser.add_argument("--all-kernels", action="store_true")
    parser.add_argument("--out-dir", default="runs/axiom-modal-b200")
    parser.add_argument("--duration-hours", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--loop", action="store_true", help="Keep cycling candidates until duration expires.")
    parser.add_argument("--once", action="store_true", help="Run one candidate per selected kernel.")
    parser.add_argument("--stream-hud", action="store_true", help="Reserved for HUD bridge; JSONL logs remain authoritative.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    kernels = list(DEFAULT_KERNELS) if args.all_kernels else [item.strip() for item in args.kernels.split(",") if item.strip()]
    summaries = []
    for kernel in kernels:
        summaries.append(optimize_kernel(args, kernel))
    print(json.dumps({"summaries": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
