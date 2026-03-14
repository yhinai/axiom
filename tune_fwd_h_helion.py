#!/usr/bin/env python3
"""
Let Helion's built-in autotuner find the best configs for chunk_fwd_h.
Uses HELION_AUTOTUNE_EFFORT=full which triggers sophisticated search:
- LFBO pattern search (PR #1115)
- DE-Surrogate hybrid (PR #1096)
- Finite search

Tries both standard Triton backend and TileIR backend (with num_ctas, occupancy).
Also tests ACF booster packs.

Usage on B200:
  # Standard backend
  python3 tune_fwd_h_helion.py
  # TileIR backend
  ENABLE_TILE=1 HELION_BACKEND=tileir python3 tune_fwd_h_helion.py
"""
import os
import sys
import time
import torch
import helion
import helion.language as hl

# Force full autotuning
os.environ["HELION_AUTOTUNE_EFFORT"] = "full"

# Benchmark shapes (scored by geomean)
BENCHMARK_SHAPES = [
    (1, 64, 1, 64, 64),
    (2, 512, 3, 64, 64),
    (2, 1024, 3, 64, 64),
]

# Test shapes (correctness only)
TEST_SHAPES = [
    (1, 64, 2, 64, 64),
    (2, 128, 4, 64, 64),
    (1, 256, 4, 64, 128),
]

# exp -> exp2 constant
_LOG2E = 1.4426950408889634


def make_args(shape):
    """Create float32 input tensors matching reference.py."""
    B, T, H, K, V = shape
    return (
        torch.randn(B, T, H, K, device="cuda", dtype=torch.float32) / K**0.5,
        torch.randn(B, T, H, K, device="cuda", dtype=torch.float32),
        torch.randn(B, T, H, V, device="cuda", dtype=torch.float32),
        torch.randn(B, T, H, device="cuda", dtype=torch.float32),
    )


def benchmark_fn(fn, args, warmup=10, iters=100):
    """Simple benchmark."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.time() - start) / iters * 1000


def autotune_shape(shape):
    """
    Let Helion autotune for a specific shape.
    Creates kernel WITHOUT config so Helion's autotuner kicks in.
    """
    B, T, H, K, V = shape
    print(f"\n{'='*60}")
    print(f"Autotuning shape: {shape}")
    print(f"Backend: {os.environ.get('HELION_BACKEND', 'triton')}")
    print(f"Effort: {os.environ.get('HELION_AUTOTUNE_EFFORT', 'default')}")
    print(f"{'='*60}")

    # Define kernel WITHOUT config — lets autotuner search
    @helion.kernel(static_shapes=True)
    def kernel(
        k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor
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
        for flat, tv in hl.tile([BH, V], block_size=[1, 8]):
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
                state = hl.dot(k[b_idx, tc, h_idx, :].T, diff_gated, acc=state, out_dtype=torch.float32)
        return h_out, v_out

    args = make_args(shape)

    try:
        # First call triggers autotuning
        print("  Starting autotuning (this may take a while)...")
        t0 = time.time()
        result = kernel(*args)
        torch.cuda.synchronize()
        tune_time = time.time() - t0
        print(f"  Autotuning took {tune_time:.1f}s")

        # Benchmark the best config
        ms = benchmark_fn(kernel, args)
        print(f"  Best time: {ms:.4f}ms")

        # Extract the best config
        cfg_repr = None
        # Try various ways to get the config
        if hasattr(kernel, '_best_config'):
            cfg_repr = repr(kernel._best_config)
        elif hasattr(kernel, '_cache'):
            for key, val in kernel._cache.items():
                if hasattr(val, '_config'):
                    cfg_repr = repr(val._config)
                    break
                if hasattr(val, 'config'):
                    cfg_repr = repr(val.config)
                    break
        # Try the bound kernel approach
        if cfg_repr is None:
            try:
                bound = kernel.bind(args)
                if hasattr(bound, '_config'):
                    cfg_repr = repr(bound._config)
                elif hasattr(bound, 'config'):
                    cfg_repr = repr(bound.config)
            except:
                pass

        if cfg_repr:
            print(f"  Config: {cfg_repr}")
        else:
            print("  (Could not extract config repr)")

        return ms, cfg_repr

    except Exception as e:
        print(f"  FAILED: {e}")
        return None, None


def autotune_shape_with_acf(shape, acf_files):
    """Try each ACF booster pack and pick the best."""
    B, T, H, K, V = shape
    print(f"\n--- ACF sweep for {shape} ---")

    best_ms = float('inf')
    best_acf = None

    for acf in acf_files:
        try:
            cfg = helion.Config(
                block_sizes=[1, 8],
                num_warps=4,
                num_stages=3,
                indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'],
                pid_type='flat',
                l2_groupings=[4],
                loop_orders=[[1, 0]],
                range_flattens=[None, True],
                range_multi_buffers=[None, False],
                range_num_stages=[0, 3],
                range_unroll_factors=[0, 0],
                range_warp_specializes=[None, None],
                load_eviction_policies=['first', '', '', '', ''],
                advanced_controls_file=acf,
            )

            @helion.kernel(static_shapes=True, config=cfg)
            def kernel(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor):
                B, T, H, K = k.shape
                V = u.shape[-1]
                C = 64
                K = hl.specialize(K)
                V = hl.specialize(V)
                NT = (T + C - 1) // C
                h_out = torch.empty(B, NT, H, K, V, dtype=k.dtype, device=k.device)
                v_out = torch.empty_like(u)
                BH = B * H
                for flat, tv in hl.tile([BH, V]):
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
                        state = hl.dot(k[b_idx, tc, h_idx, :].T, diff_gated, acc=state, out_dtype=torch.float32)
                return h_out, v_out

            args = make_args(shape)
            result = kernel(*args)
            ms = benchmark_fn(kernel, args, warmup=5, iters=50)

            acf_name = os.path.basename(acf)
            if ms < best_ms:
                best_ms = ms
                best_acf = acf
                print(f"  {acf_name}: {ms:.4f}ms *** BEST")
            else:
                print(f"  {acf_name}: {ms:.4f}ms")

        except Exception as e:
            acf_name = os.path.basename(acf)
            print(f"  {acf_name}: FAILED ({str(e)[:80]})")

    return best_ms, best_acf


def main():
    import glob
    import math

    mode = sys.argv[1] if len(sys.argv) > 1 else "autotune"
    backend = os.environ.get('HELION_BACKEND', 'triton')

    if mode == "autotune":
        # Let Helion autotune each benchmark shape
        shapes = BENCHMARK_SHAPES
        if len(sys.argv) > 2 and sys.argv[2] == "all":
            shapes = BENCHMARK_SHAPES + TEST_SHAPES

        results = {}
        for shape in shapes:
            ms, cfg = autotune_shape(shape)
            if ms is not None:
                results[shape] = (ms, cfg)

        # Summary
        print(f"\n\n{'='*60}")
        print(f"AUTOTUNE SUMMARY (backend={backend})")
        print(f"{'='*60}")
        bench_times = []
        for shape, (ms, cfg) in results.items():
            is_bench = shape in BENCHMARK_SHAPES
            marker = " [BENCH]" if is_bench else ""
            print(f"\n  {shape}{marker}: {ms:.4f}ms")
            if cfg:
                print(f"    {cfg}")
            if is_bench:
                bench_times.append(ms)

        if bench_times:
            geomean = math.exp(sum(math.log(x) for x in bench_times) / len(bench_times))
            print(f"\n  Benchmark geomean: {geomean:.4f}ms = {geomean/1000:.6e}s")

    elif mode == "acf":
        # Try ACF booster packs
        acf_files = sorted(glob.glob("/opt/booster_pack/chunk_fwd_h_*.acf"))
        if not acf_files:
            # Try all ACFs
            acf_files = sorted(glob.glob("/opt/booster_pack/*.acf"))
        print(f"Found {len(acf_files)} ACF files")

        for shape in BENCHMARK_SHAPES:
            autotune_shape_with_acf(shape, acf_files)

    elif mode == "tileir":
        # Force TileIR backend
        os.environ["ENABLE_TILE"] = "1"
        os.environ["HELION_BACKEND"] = "tileir"
        for shape in BENCHMARK_SHAPES:
            autotune_shape(shape)


if __name__ == "__main__":
    main()
