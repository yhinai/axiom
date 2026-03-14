#!/usr/bin/env python3
"""
Tune FP8 quant kernel using eval.py's exact benchmarking methodology.
Tests config variations by modifying submission.py and re-running eval.py benchmark.
"""
import subprocess
import sys
import re
import os
import json

SUBMISSION = "fp8_quant_py/submission.py"
EVAL_CMD = [sys.executable, "eval.py", "benchmark", "fp8_quant_py/"]

def read_submission():
    with open(SUBMISSION) as f:
        return f.read()

def write_submission(content):
    with open(SUBMISSION, "w") as f:
        f.write(content)

def run_benchmark():
    """Run eval.py benchmark and extract times."""
    result = subprocess.run(EVAL_CMD, capture_output=True, text=True, timeout=120)
    times = []
    for line in result.stdout.split("\n"):
        m = re.search(r"Benchmark \d+: ([\d.]+) ms", line)
        if m:
            times.append(float(m.group(1)))
    return times

def make_config_str(bs, nw, ns):
    return f"helion.Config(block_sizes=[{bs}], num_warps={nw}, num_stages={ns})"

# Current baseline configs
baseline_configs = {
    (256, 4096, 128): (64, 4, 2),
    (256, 8192, 128): (64, 4, 2),
    (4096, 7168, 128): (128, 8, 2),
}

# Save original
original = read_submission()

# Run baseline
print("=== BASELINE ===")
baseline_times = run_benchmark()
print(f"  Times: {baseline_times}")

# Try variations for each benchmark shape
shapes = list(baseline_configs.keys())
best_configs = dict(baseline_configs)
best_times = list(baseline_times)

variations = [
    # (block_sizes, num_warps, num_stages)
    (32, 4, 1), (32, 4, 2), (32, 4, 3),
    (32, 8, 1), (32, 8, 2), (32, 8, 3),
    (64, 4, 1), (64, 4, 3), (64, 4, 4),
    (64, 8, 1), (64, 8, 2), (64, 8, 3), (64, 8, 4),
    (64, 16, 1), (64, 16, 2), (64, 16, 3),
    (128, 4, 1), (128, 4, 2), (128, 4, 3),
    (128, 8, 1), (128, 8, 3), (128, 8, 4),
    (128, 16, 1), (128, 16, 2), (128, 16, 3),
    (256, 4, 1), (256, 4, 2), (256, 4, 3),
    (256, 8, 1), (256, 8, 2), (256, 8, 3),
    (256, 16, 1), (256, 16, 2), (256, 16, 3),
]

for shape_idx, shape in enumerate(shapes):
    T, H, gsz = shape
    print(f"\n=== Shape ({T}, {H}, {gsz}) — baseline: {best_times[shape_idx]:.4f}ms ===")

    for bs, nw, ns in variations:
        # Modify only this shape's config
        content = original
        old_cfg = make_config_str(*baseline_configs[shape])
        new_cfg = make_config_str(bs, nw, ns)
        old_line = f"({T}, {H}, {gsz}): {old_cfg},"
        new_line = f"({T}, {H}, {gsz}): {new_cfg},"
        content = content.replace(old_line, new_line)
        write_submission(content)

        try:
            times = run_benchmark()
            if times and len(times) > shape_idx:
                ms = times[shape_idx]
                if ms < best_times[shape_idx]:
                    print(f"  *** bs={bs} w={nw} s={ns}: {ms:.4f}ms (was {best_times[shape_idx]:.4f}ms)")
                    best_times[shape_idx] = ms
                    best_configs[shape] = (bs, nw, ns)
        except Exception as e:
            pass

    # Reset to original + best config for this shape
    bs, nw, ns = best_configs[shape]
    old_cfg = make_config_str(*baseline_configs[shape])
    new_cfg = make_config_str(bs, nw, ns)
    old_line = f"({T}, {H}, {gsz}): {old_cfg},"
    new_line = f"({T}, {H}, {gsz}): {new_cfg},"
    original = original.replace(old_line, new_line)

# Write final best configs
write_submission(original)

print("\n=== FINAL RESULTS ===")
final_times = run_benchmark()
for i, shape in enumerate(shapes):
    print(f"  {shape}: {baseline_times[i]:.4f}ms -> {final_times[i]:.4f}ms ({(1 - final_times[i]/baseline_times[i])*100:+.1f}%)")

print("\n=== BEST CONFIGS ===")
for shape, (bs, nw, ns) in best_configs.items():
    print(f"  {shape}: bs={bs}, w={nw}, s={ns}")
