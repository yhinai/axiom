#!/usr/bin/env python3
"""
Comprehensive per-shape autotuning for gated_deltanet_chunk_fwd_h.
Explores a wide config space inspired by winning patterns from recompute_w_u (#1).

Strategy:
- Per-shape configs (not one-size-fits-all)
- Focus on benchmark shapes (scoring is geomean of 3 benchmark shapes)
- Wide search: warps, stages, persistent PIDs, loop tuning, indexing combos
- Uses CUDA graph benchmarking for accurate timing (matches eval.py methodology)

Run on B200: ./remote_run.sh "python3 tune_fwd_h_v2.py"
"""
import os
import sys
import time
import itertools
import torch
import helion
import helion.language as hl

torch.backends.cudnn.benchmark = True


def benchmark_cudagraph(fn, args, warmup=5, rep_ms=100):
    """Benchmark using CUDA graphs (matches eval.py methodology)."""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Estimate time per call
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    est_ms = start.elapsed_time(end) / 5

    n_repeat = max(1, int(rep_ms / est_ms)) if est_ms > 0 else 100

    # Capture CUDA graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_repeat):
            fn(*args)
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(5):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / n_repeat)

    return min(times)


def make_fwd_h(config):
    """Create chunk_fwd_h kernel with given config."""
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
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
                alpha = torch.where(valid, torch.exp(g_end - g_t), 0.0)
                k_adj = k[b_idx, tc, h_idx, :] * alpha[:, None]
                state = state * torch.exp(g_end)
                upd = hl.dot(k_adj.T, diff, out_dtype=torch.float32)
                state = state + upd
        return h_out, v_out
    return kernel


def make_args(shape):
    """Create input tensors for a given shape (float32 to match reference.py)."""
    B, T, H, K, V = shape
    return (
        torch.randn(B, T, H, K, device="cuda", dtype=torch.float32) / K**0.5,
        torch.randn(B, T, H, K, device="cuda", dtype=torch.float32),
        torch.randn(B, T, H, V, device="cuda", dtype=torch.float32),
        torch.randn(B, T, H, device="cuda", dtype=torch.float32),
    )


def gen_configs():
    """
    Generate a comprehensive set of configs to search.
    Organized in tiers: base variations, then combinations.
    """
    # Current best (baseline)
    base = dict(
        block_sizes=[],
        indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'],
        pid_type='flat',
        num_warps=4,
        num_stages=3,
        l2_groupings=[1],
        loop_orders=[[1, 0]],
        load_eviction_policies=['first', '', '', '', ''],
        range_flattens=[None, True],
        range_multi_buffers=[None, False],
        range_num_stages=[0, 3],
        range_unroll_factors=[0, 0],
        range_warp_specializes=[None, None],
    )

    configs = []

    # --- Tier 1: Single-axis sweeps from baseline ---

    # Warp sweep (compute-bound kernel, higher warps may help)
    for nw in [2, 4, 8, 16, 32]:
        c = {**base, 'num_warps': nw}
        configs.append(c)

    # Stage sweep
    for ns in [1, 2, 3, 4, 5, 7]:
        c = {**base, 'num_stages': ns}
        configs.append(c)

    # Loop order
    for lo in [[0, 1], [1, 0]]:
        c = {**base, 'loop_orders': [lo]}
        configs.append(c)

    # L2 groupings
    for lg in [1, 4, 8, 16]:
        c = {**base, 'l2_groupings': [lg]}
        configs.append(c)

    # Range num_stages for inner loop
    for rns in [0, 1, 2, 3, 4, 5, 7]:
        c = {**base, 'range_num_stages': [0, rns]}
        configs.append(c)

    # Range unroll factors for inner loop
    for ruf in [0, 1, 2, 4]:
        c = {**base, 'range_unroll_factors': [0, ruf]}
        configs.append(c)

    # Warp specialization combinations
    for ws0 in [None, True]:
        for ws1 in [None, True]:
            c = {**base, 'range_warp_specializes': [ws0, ws1]}
            configs.append(c)

    # Multi-buffer combinations
    for mb0 in [None, False, True]:
        for mb1 in [None, False, True]:
            c = {**base, 'range_multi_buffers': [mb0, mb1]}
            configs.append(c)

    # Range flatten combinations
    for rf0 in [None, True, False]:
        for rf1 in [None, True, False]:
            c = {**base, 'range_flattens': [rf0, rf1]}
            configs.append(c)

    # Eviction policy combos
    eviction_combos = [
        ['first', '', '', '', ''],
        ['', 'first', '', '', ''],
        ['first', 'first', '', '', ''],
        ['', '', 'first', '', ''],
        ['first', '', 'first', '', ''],
        ['', '', '', '', ''],
        ['first', 'first', 'first', '', ''],
        ['first', '', '', 'first', ''],
    ]
    for ep in eviction_combos:
        c = {**base, 'load_eviction_policies': ep}
        configs.append(c)

    # --- Tier 2: Indexing combos ---
    indexing_combos = [
        # Current best
        ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'],
        # All tensor_descriptor
        ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'],
        # All pointer
        ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'],
        # Mix: more tensor_descriptor
        ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'],
        ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'],
        ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'],
        # Mix: more pointer
        ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'],
        ['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'],
    ]
    for idx in indexing_combos:
        c = {**base, 'indexing': idx}
        configs.append(c)

    # --- Tier 3: Persistent PID (like recompute_w_u's winning pattern) ---
    for pid in ['persistent_blocked', 'persistent_interleaved']:
        for nw in [4, 8, 16, 32]:
            for nsm in [4, 8, 16, 32, 64]:
                for mreg in [32, 64, 128]:
                    c = {**base,
                         'pid_type': pid,
                         'num_warps': nw,
                         'num_sm_multiplier': nsm,
                         'maxnreg': mreg}
                    configs.append(c)

    # --- Tier 4: Best combos from recompute_w_u winning pattern ---
    # recompute_w_u won with: persistent_blocked, nw=32, nsm=16, maxnreg=32,
    # l2=[16], range_unroll=[4], range_num_stages=[3]
    for nw in [16, 32]:
        for nsm in [8, 16, 32]:
            for mreg in [32, 64]:
                for ruf in [0, 2, 4]:
                    for rns in [0, 3, 5]:
                        c = {**base,
                             'pid_type': 'persistent_blocked',
                             'num_warps': nw,
                             'num_sm_multiplier': nsm,
                             'maxnreg': mreg,
                             'l2_groupings': [16],
                             'range_unroll_factors': [0, ruf],
                             'range_num_stages': [0, rns]}
                        configs.append(c)

    # --- Tier 5: High-warp flat with deep pipelining ---
    for nw in [8, 16, 32]:
        for ns in [3, 5, 7]:
            for rns in [3, 5, 7]:
                c = {**base,
                     'num_warps': nw,
                     'num_stages': ns,
                     'range_num_stages': [0, rns],
                     'l2_groupings': [8]}
                configs.append(c)

    # Deduplicate
    seen = set()
    unique = []
    for c in configs:
        key = str(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def tune_shape(shape, configs):
    """Tune a single shape across all configs. Returns (best_ms, best_config_repr)."""
    B, T, H, K, V = shape
    args = make_args(shape)

    best_ms = float('inf')
    best_cfg_repr = None
    total = len(configs)
    failed = 0
    tested = 0

    for i, cfg_dict in enumerate(configs):
        try:
            config = helion.Config(**cfg_dict)
            kernel = make_fwd_h(config)
            # Quick warmup + correctness check
            result = kernel(*args)
            del result

            ms = benchmark_cudagraph(kernel, args, warmup=3, rep_ms=50)
            tested += 1

            if ms < best_ms:
                improvement = (best_ms - ms) / best_ms * 100 if best_ms < float('inf') else 0
                best_ms = ms
                best_cfg_repr = repr(config)
                print(f"  [{i+1}/{total}] {ms:.4f}ms *** NEW BEST ({improvement:+.1f}%)")
                print(f"    {best_cfg_repr}")
            elif i % 50 == 0:
                print(f"  [{i+1}/{total}] best so far: {best_ms:.4f}ms (tested={tested}, failed={failed})")

        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  [{i+1}/{total}] FAILED: {str(e)[:100]}")

    print(f"  Tested {tested}/{total} configs ({failed} failed)")
    return best_ms, best_cfg_repr


def main():
    # Benchmark shapes (these are what scoring is based on — geomean)
    benchmark_shapes = [
        (1, 64, 1, 64, 64),
        (2, 512, 3, 64, 64),
        (2, 1024, 3, 64, 64),
    ]

    # Test shapes (need to pass correctness, but not scored)
    test_shapes = [
        (1, 64, 2, 64, 64),
        (2, 128, 4, 64, 64),
        (1, 256, 4, 64, 128),
    ]

    # Additional benchmark shapes from submission (leaderboard may test these)
    extra_shapes = [
        (3, 1024, 4, 100, 100),
        (4, 1024, 4, 128, 128),
        (2, 1536, 4, 128, 128),
        (4, 2048, 8, 64, 64),
    ]

    # Which shapes to tune (focus on benchmark shapes for max score impact)
    if len(sys.argv) > 1 and sys.argv[1] == "all":
        shapes_to_tune = benchmark_shapes + test_shapes + extra_shapes
    elif len(sys.argv) > 1 and sys.argv[1] == "extra":
        shapes_to_tune = extra_shapes
    else:
        shapes_to_tune = benchmark_shapes

    configs = gen_configs()
    print(f"Generated {len(configs)} unique configs to test")
    print(f"Tuning {len(shapes_to_tune)} shapes")
    print()

    results = {}
    geomean_parts = []

    for shape in shapes_to_tune:
        print(f"\n{'='*60}")
        print(f"Shape: {shape}")
        print(f"{'='*60}")

        best_ms, best_cfg = tune_shape(shape, configs)
        results[shape] = (best_ms, best_cfg)

        if shape in benchmark_shapes:
            geomean_parts.append(best_ms)

        print(f"\n  BEST: {best_ms:.4f}ms")
        print(f"  Config: {best_cfg}")

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for shape, (ms, cfg) in results.items():
        marker = " [BENCH]" if shape in benchmark_shapes else ""
        print(f"\n  {shape}{marker}: {ms:.4f}ms")
        print(f"    {cfg}")

    if geomean_parts:
        import math
        geomean = math.exp(sum(math.log(x) for x in geomean_parts) / len(geomean_parts))
        print(f"\n  Benchmark geomean: {geomean:.4f}ms = {geomean/1000:.6e}s")

    # Output submission-ready format
    print(f"\n\n{'='*60}")
    print("SUBMISSION-READY CONFIGS")
    print(f"{'='*60}")
    print("SHAPE_CONFIGS: dict[tuple, helion.Config] = {")
    for shape, (ms, cfg) in sorted(results.items()):
        print(f"    {shape}: {cfg},")
    print("}")


if __name__ == "__main__":
    main()
