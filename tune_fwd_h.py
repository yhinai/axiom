#!/usr/bin/env python3
"""
Per-shape autotuning for chunk_fwd_h using Helion's built-in autotuner.

Usage (on B200 server):
  HELION_AUTOTUNE_EFFORT=full python tune_fwd_h.py           # benchmark shapes only
  HELION_AUTOTUNE_EFFORT=full python tune_fwd_h.py all       # all shapes
  HELION_AUTOTUNE_EFFORT=full python tune_fwd_h.py 1,64,1,64,64  # single shape

The script creates a fresh kernel per shape (no config = triggers autotuner),
benchmarks the result, and prints the best config ready for submission.py.
"""
import os
import sys
import time
import torch
import helion
import helion.language as hl

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "full")

_LOG2E = 1.4426950408889634


def make_kernel(config=None):
    """Create kernel. If config=None, Helion autotuner searches for the best."""
    kwargs = dict(static_shapes=True, dot_precision="tf32")
    if config is not None:
        kwargs["config"] = config

    @helion.kernel(**kwargs)
    def kernel(
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = u.shape[-1]
        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)
        NT = (T + C - 1) // C
        h_out = torch.empty(B, NT, H, K, V, dtype=k.dtype, device=k.device)
        v_out = torch.empty_like(u)
        BH = B * H
        for flat, tv in hl.tile([BH, V], block_size=[1, None]):
            b_idx = flat.begin // H
            h_idx = flat.begin % H
            state = hl.zeros([K, tv], dtype=torch.float32)
            for tc in hl.tile(T, block_size=C):
                chunk_idx = tc.begin // C
                t_end = min(tc.begin + C, T) - 1
                h_out[b_idx, chunk_idx, h_idx, :, tv] = state.to(k.dtype)
                proj = hl.dot(w[b_idx, tc, h_idx, :], state, out_dtype=torch.float32)
                diff = u[b_idx, tc, h_idx, tv].to(torch.float32) - proj
                v_out[b_idx, tc, h_idx, tv] = diff.to(u.dtype)
                g_end = g[b_idx, t_end, h_idx].to(torch.float32)
                g_t = g[b_idx, tc, h_idx].to(torch.float32)
                valid = tc.index < T
                alpha = torch.where(valid, torch.exp2((g_end - g_t) * _LOG2E), 0.0)
                diff_gated = diff * alpha[:, None]
                state = state * torch.exp2(g_end * _LOG2E)
                state = hl.dot(
                    k[b_idx, tc, h_idx, :].T, diff_gated, acc=state, out_dtype=torch.float32
                )
        return h_out, v_out
    return kernel


def make_args(shape):
    B, T, H, K, V = shape
    k = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32)
    w = torch.randn(B, T, H, K, device="cuda", dtype=torch.float32)
    u = torch.randn(B, T, H, V, device="cuda", dtype=torch.float32)
    g = torch.randn(B, T, H, device="cuda", dtype=torch.float32).cumsum(1) * -0.1
    return k, w, u, g


def benchmark(kernel, args, warmup=30, iters=300):
    for _ in range(warmup):
        kernel(*args)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        kernel(*args)
    torch.cuda.synchronize()
    return (time.time() - start) / iters * 1000


BENCHMARK_SHAPES = [
    (1, 64, 1, 64, 64),
    (2, 512, 3, 64, 64),
    (2, 1024, 3, 64, 64),
]

TEST_SHAPES = [
    (1, 64, 2, 64, 64),
    (2, 128, 4, 64, 64),
    (1, 256, 4, 64, 128),
]


def tune_shape(shape):
    """Autotune a single shape and return (ms, config_repr)."""
    B, T, H, K, V = shape
    print(f"\n{'='*60}")
    print(f"Shape: {shape}  BH={B*H}  chunks={T//64}  V={V}")
    print(f"{'='*60}")

    args = make_args(shape)

    # Let Helion autotuner find the best config
    print("Running Helion autotuner...")
    kernel = make_kernel(config=None)
    try:
        result = kernel(*args)
    except Exception as e:
        print(f"  Autotuner failed: {e}")
        return None, None

    ms = benchmark(kernel, args)
    print(f"  Autotuned time: {ms:.4f}ms")

    # Try to extract config
    config_repr = None
    for attr in ["config", "best_config", "_config", "_best_config"]:
        cfg = getattr(kernel, attr, None)
        if cfg is not None:
            config_repr = repr(cfg)
            break

    if config_repr:
        print(f"  Config: {config_repr}")
    else:
        print("  (Could not extract config — check kernel._compiled_fn or cache)")
        # Try deeper inspection
        for attr in dir(kernel):
            if "config" in attr.lower() or "cache" in attr.lower():
                val = getattr(kernel, attr, None)
                if val is not None and not callable(val):
                    print(f"    kernel.{attr} = {val}")

    return ms, config_repr


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        shapes = TEST_SHAPES + BENCHMARK_SHAPES
    elif len(sys.argv) > 1 and "," in sys.argv[1]:
        # Parse single shape like "1,64,1,64,64"
        parts = tuple(int(x) for x in sys.argv[1].split(","))
        shapes = [parts]
    else:
        shapes = BENCHMARK_SHAPES

    effort = os.environ.get("HELION_AUTOTUNE_EFFORT", "default")
    print(f"Autotuning chunk_fwd_h — {len(shapes)} shapes, effort={effort}")

    results = {}
    for shape in shapes:
        ms, cfg = tune_shape(shape)
        if ms is not None:
            results[shape] = (ms, cfg)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY — paste into submission.py")
    print(f"{'='*60}")
    for shape, (ms, cfg) in results.items():
        print(f"# {shape}: {ms:.4f}ms")
        if cfg:
            print(f"# ({shape}): {cfg},")
        print()

    if len(results) >= 3:
        benchmark_times = [results[s][0] for s in BENCHMARK_SHAPES if s in results]
        if benchmark_times:
            geomean = 1.0
            for t in benchmark_times:
                geomean *= t
            geomean = geomean ** (1.0 / len(benchmark_times))
            print(f"Geomean (benchmark shapes): {geomean:.4f}ms")
