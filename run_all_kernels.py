#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


KERNEL_DIRS = [
    "causal_conv1d_py",
    "gated_deltanet_chunk_fwd_h_py",
    "gated_deltanet_chunk_fwd_o_py",
    "gated_deltanet_recompute_w_u_py",
]

BENCHMARK_RE = re.compile(
    r"^\s*Benchmark\s+\d+:\s+"
    r"(?P<mean>[0-9.]+)\s+ms\s+"
    r"\(min=(?P<min>[0-9.]+),\s+max=(?P<max>[0-9.]+)\)\s+"
    r"(?P<spec>.+)$"
)
TEST_RE = re.compile(r"^\s*Test\s+\d+:\s+(?P<status>PASS|FAIL)\b")


@dataclass
class BenchmarkResult:
    mean_ms: float
    min_ms: float
    max_ms: float
    spec: str


@dataclass
class KernelSummary:
    name: str
    tests_passed: bool
    benchmark_failures: int
    fastest: BenchmarkResult | None
    return_code: int


def ms_to_us_text(value_ms: float) -> str:
    return f"{value_ms * 1000:.1f} us"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all Helion kernels and report the fastest benchmark per kernel."
    )
    parser.add_argument(
        "--mode",
        choices=("both", "test", "benchmark"),
        default="both",
        help="Which eval.py mode to run for each kernel.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for eval.py. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running later kernels even if an earlier kernel fails.",
    )
    return parser.parse_args()


def run_kernel(repo_root: Path, python_exe: str, mode: str, kernel_dir: str) -> KernelSummary:
    cmd = [python_exe, "eval.py", mode, f"{kernel_dir}/"]
    process = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    tests_passed = True
    benchmark_failures = 0
    benchmarks: list[BenchmarkResult] = []

    print(f"\n=== {kernel_dir} ===")
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")

        benchmark_match = BENCHMARK_RE.match(line)
        if benchmark_match:
            result = BenchmarkResult(
                mean_ms=float(benchmark_match.group("mean")),
                min_ms=float(benchmark_match.group("min")),
                max_ms=float(benchmark_match.group("max")),
                spec=benchmark_match.group("spec"),
            )
            benchmarks.append(result)
            print(
                re.sub(
                    r"(?P<mean>[0-9.]+)\s+ms\s+\(min=(?P<min>[0-9.]+),\s+max=(?P<max>[0-9.]+)\)",
                    (
                        f"{ms_to_us_text(result.mean_ms)} "
                        f"(min={ms_to_us_text(result.min_ms)}, max={ms_to_us_text(result.max_ms)})"
                    ),
                    line,
                    count=1,
                )
            )
        else:
            print(line)

        test_match = TEST_RE.match(line)
        if test_match and test_match.group("status") == "FAIL":
            tests_passed = False

        if "Some tests FAILED." in line:
            tests_passed = False

        if "FAIL (correctness)" in line:
            benchmark_failures += 1

    return_code = process.wait()
    fastest = min(benchmarks, key=lambda result: result.min_ms) if benchmarks else None
    return KernelSummary(
        name=kernel_dir,
        tests_passed=tests_passed,
        benchmark_failures=benchmark_failures,
        fastest=fastest,
        return_code=return_code,
    )


def print_summary(summaries: list[KernelSummary], mode: str) -> None:
    print("\n=== Summary ===")
    for summary in summaries:
        status = "PASS" if summary.return_code == 0 else "FAIL"
        if mode == "test":
            print(f"{summary.name}: {status} tests")
            continue

        if summary.fastest is None:
            print(f"{summary.name}: {status} no benchmark results")
            continue

        print(
            f"{summary.name}: {status} fastest min={ms_to_us_text(summary.fastest.min_ms)} "
            f"(mean={ms_to_us_text(summary.fastest.mean_ms)}, "
            f"max={ms_to_us_text(summary.fastest.max_ms)}) "
            f"for {summary.fastest.spec}"
        )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    summaries: list[KernelSummary] = []

    for kernel_dir in KERNEL_DIRS:
        summary = run_kernel(repo_root, args.python, args.mode, kernel_dir)
        summaries.append(summary)
        if summary.return_code != 0 and not args.keep_going:
            break

    print_summary(summaries, args.mode)
    return 0 if summaries and all(summary.return_code == 0 for summary in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
