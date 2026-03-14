#!/usr/bin/env python3
"""
Optimization sweep for gated_deltanet_chunk_fwd_h kernel on B200.
Tests various config axes on shape (2, 1024, 3, 64, 64).
"""

import torch
import helion
import helion.language as hl
import time
import traceback
import sys

_LOG2E = 1.4426950408889634

def make_kernel(config, dot_precision="tf32"):
    @helion.kernel(static_shapes=True, dot_precision=dot_precision, config=config)
    def kernel(k, w, u, g):
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
                state = hl.dot(k[b_idx, tc, h_idx, :].T, diff_gated, acc=state, out_dtype=torch.float32)
        return h_out, v_out
    return kernel


def benchmark_kernel(kernel_fn, k, w, u, g, warmup=5, iters=20):
    """Benchmark a kernel, return median time in seconds."""
    for _ in range(warmup):
        kernel_fn(k, w, u, g)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        kernel_fn(k, w, u, g)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    return times[len(times) // 2]  # median


def test_config(config, dot_precision, k, w, u, g, label=""):
    """Test a single config, return (label, time_s) or (label, None) on failure."""
    try:
        kernel_fn = make_kernel(config, dot_precision=dot_precision)
        result = kernel_fn(k, w, u, g)
        torch.cuda.synchronize()
        t = benchmark_kernel(kernel_fn, k, w, u, g)
        return (label, t, config, dot_precision)
    except Exception as e:
        err_str = str(e)[:120]
        print(f"  FAIL [{label}]: {err_str}", flush=True)
        return (label, None, config, dot_precision)


def make_config(block_sizes=[8], num_warps=4, num_stages=3, l2_groupings=[4],
                loop_orders=[[1, 0]], indexing=None, load_eviction_policies=None,
                range_num_stages=[0, 3], range_unroll_factors=[0, 0],
                range_flattens=[None, True], range_multi_buffers=[None, False],
                pid_type='flat', range_warp_specializes=[None, None]):
    if indexing is None:
        indexing = ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer']
    if load_eviction_policies is None:
        load_eviction_policies = ['first', '', '', '', '']
    return helion.Config(
        block_sizes=block_sizes,
        indexing=indexing,
        l2_groupings=l2_groupings,
        load_eviction_policies=load_eviction_policies,
        loop_orders=loop_orders,
        num_stages=num_stages,
        num_warps=num_warps,
        pid_type=pid_type,
        range_flattens=range_flattens,
        range_multi_buffers=range_multi_buffers,
        range_num_stages=range_num_stages,
        range_unroll_factors=range_unroll_factors,
        range_warp_specializes=range_warp_specializes,
    )


def main():
    torch.manual_seed(2146)
    device = "cuda"

    B, T, H, K, V = 2, 1024, 3, 64, 64
    k = torch.randn(B, T, H, K, device=device) / K**0.5
    w = torch.randn(B, T, H, K, device=device)
    u = torch.randn(B, T, H, V, device=device)
    g = torch.randn(B, T, H, device=device)

    print(f"Shape: B={B}, T={T}, H={H}, K={K}, V={V}", flush=True)
    print(f"Device: {torch.cuda.get_device_name()}", flush=True)
    print("=" * 80, flush=True)

    results = []

    # =========================================================================
    # Current best config as baseline
    # =========================================================================
    print("\n=== BASELINE (current best, l2_groupings=[4]) ===", flush=True)
    cfg = make_config(l2_groupings=[4])
    r = test_config(cfg, "tf32", k, w, u, g, label="baseline_l2g4")
    results.append(r)
    if r[1]:
        print(f"  baseline_l2g4: {r[1]*1e6:.1f} us", flush=True)

    cfg = make_config(l2_groupings=[1])
    r = test_config(cfg, "tf32", k, w, u, g, label="baseline_l2g1")
    results.append(r)
    if r[1]:
        print(f"  baseline_l2g1: {r[1]*1e6:.1f} us", flush=True)

    # =========================================================================
    # 1. DOT PRECISION: ieee vs tf32
    # =========================================================================
    print("\n=== DOT PRECISION ===", flush=True)
    for dp in ["ieee", "tf32"]:
        cfg = make_config()
        label = f"dot_precision={dp}"
        r = test_config(cfg, dp, k, w, u, g, label=label)
        results.append(r)
        if r[1]:
            print(f"  {label}: {r[1]*1e6:.1f} us", flush=True)

    # =========================================================================
    # 2. NUM_WARPS sweep
    # =========================================================================
    print("\n=== NUM_WARPS ===", flush=True)
    for nw in [2, 4, 8, 16, 32]:
        cfg = make_config(num_warps=nw)
        label = f"num_warps={nw}"
        r = test_config(cfg, "tf32", k, w, u, g, label=label)
        results.append(r)
        if r[1]:
            print(f"  {label}: {r[1]*1e6:.1f} us", flush=True)

    # =========================================================================
    # 3. NUM_STAGES sweep
    # =========================================================================
    print("\n=== NUM_STAGES ===", flush=True)
    for ns in [1, 2, 3, 4, 5, 7]:
        cfg = make_config(num_stages=ns)
        label = f"num_stages={ns}"
        r = test_config(cfg, "tf32", k, w, u, g, label=label)
        results.append(r)
        if r[1]:
            print(f"  {label}: {r[1]*1e6:.1f} us", flush=True)

    # =========================================================================
    # 4. L2_GROUPINGS sweep
    # =========================================================================
    print("\n=== L2_GROUPINGS ===", flush=True)
    for lg in [1, 2, 4, 8, 16]:
        cfg = make_config(l2_groupings=[lg])
        label = f"l2_groupings=[{lg}]"
        r = test_config(cfg, "tf32", k, w, u, g, label=label)
        results.append(r)
        if r[1]:
            print(f"  {label}: {r[1]*1e6:.1f} us", flush=True)

    # =========================================================================
    # 5. LOOP_ORDERS
    # =========================================================================
    print("\n=== LOOP_ORDERS ===", flush=True)
    for lo in [[0, 1], [1, 0]]:
        cfg = make_config(loop_orders=[lo])
        label = f"loop_orders=[{lo}]"
        r = test_config(cfg, "tf32", k, w, u, g, label=label)
        results.append(r)
        if r[1]:
            print(f"  {label}: {r[1]*1e6:.1f} us", flush=True)

    # =========================================================================
    # 6. RANGE_NUM_STAGES for inner loop
    # =========================================================================
    print("\n=== RANGE_NUM_STAGES (inner loop, index 1) ===", flush=True)
    for rns in [0, 1, 2, 3, 5]:
        cfg = make_config(range_num_stages=[0, rns])
        label = f"range_num_stages=[0,{rns}]"
        r = test_config(cfg, "tf32", k, w, u, g, label=label)
        results.append(r)
        if r[1]:
            print(f"  {label}: {r[1]*1e6:.1f} us", flush=True)

    # =========================================================================
    # 7. RANGE_UNROLL_FACTORS for inner loop
    # =========================================================================
    print("\n=== RANGE_UNROLL_FACTORS (inner loop, index 1) ===", flush=True)
    for ruf in [0, 1, 2, 4]:
        cfg = make_config(range_unroll_factors=[0, ruf])
        label = f"range_unroll_factors=[0,{ruf}]"
        r = test_config(cfg, "tf32", k, w, u, g, label=label)
        results.append(r)
        if r[1]:
            print(f"  {label}: {r[1]*1e6:.1f} us", flush=True)

    # =========================================================================
    # 8. INDEXING combos
    # =========================================================================
    print("\n=== INDEXING COMBOS ===", flush=True)
    indexing_combos = {
        "all_pointer": ['pointer'] * 7,
        "all_tensor_desc": ['tensor_descriptor'] * 7,
        "current_mixed": ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'],
        "mixed_v2": ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'],
        "mixed_v3": ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'],
        "mixed_v4": ['pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'],
    }
    for name, idx in indexing_combos.items():
        cfg = make_config(indexing=idx)
        label = f"indexing={name}"
        r = test_config(cfg, "tf32", k, w, u, g, label=label)
        results.append(r)
        if r[1]:
            print(f"  {label}: {r[1]*1e6:.1f} us", flush=True)

    # =========================================================================
    # 9. EVICTION POLICIES
    # =========================================================================
    print("\n=== EVICTION POLICIES ===", flush=True)
    eviction_combos = {
        "all_empty": ['', '', '', '', ''],
        "all_first": ['first', 'first', 'first', 'first', 'first'],
        "all_last": ['last', 'last', 'last', 'last', 'last'],
        "current": ['first', '', '', '', ''],
        "first_last_mix": ['first', 'last', '', 'first', ''],
        "last_first_mix": ['last', 'first', '', '', 'last'],
        "first_empty_last": ['first', '', 'last', '', 'first'],
    }
    for name, ep in eviction_combos.items():
        cfg = make_config(load_eviction_policies=ep)
        label = f"eviction={name}"
        r = test_config(cfg, "tf32", k, w, u, g, label=label)
        results.append(r)
        if r[1]:
            print(f"  {label}: {r[1]*1e6:.1f} us", flush=True)

    # =========================================================================
    # 10. BLOCK_SIZES for V tile dimension
    # =========================================================================
    print("\n=== BLOCK_SIZES (V tile) ===", flush=True)
    for bs in [8, 16, 32, 64]:
        cfg = make_config(block_sizes=[bs])
        label = f"block_sizes=[{bs}]"
        r = test_config(cfg, "tf32", k, w, u, g, label=label)
        results.append(r)
        if r[1]:
            print(f"  {label}: {r[1]*1e6:.1f} us", flush=True)

    # =========================================================================
    # 11. PID_TYPE
    # =========================================================================
    print("\n=== PID_TYPE ===", flush=True)
    for pt in ['flat', 'linear', 'persistent_blocked']:
        cfg = make_config(pid_type=pt)
        label = f"pid_type={pt}"
        r = test_config(cfg, "tf32", k, w, u, g, label=label)
        results.append(r)
        if r[1]:
            print(f"  {label}: {r[1]*1e6:.1f} us", flush=True)

    # =========================================================================
    # 12. RANGE_MULTI_BUFFERS
    # =========================================================================
    print("\n=== RANGE_MULTI_BUFFERS ===", flush=True)
    for rmb in [False, True]:
        cfg = make_config(range_multi_buffers=[None, rmb])
        label = f"range_multi_buffers=[None,{rmb}]"
        r = test_config(cfg, "tf32", k, w, u, g, label=label)
        results.append(r)
        if r[1]:
            print(f"  {label}: {r[1]*1e6:.1f} us", flush=True)

    # =========================================================================
    # 13. RANGE_FLATTENS
    # =========================================================================
    print("\n=== RANGE_FLATTENS ===", flush=True)
    for rf in [True, False]:
        cfg = make_config(range_flattens=[None, rf])
        label = f"range_flattens=[None,{rf}]"
        r = test_config(cfg, "tf32", k, w, u, g, label=label)
        results.append(r)
        if r[1]:
            print(f"  {label}: {r[1]*1e6:.1f} us", flush=True)

    # =========================================================================
    # 14. COMBINED: targeted cross-product of most impactful params
    # =========================================================================
    print("\n=== COMBINED SWEEP (targeted cross-product) ===", flush=True)
    combined_count = 0
    for nw in [4, 8, 16]:
        for ns in [2, 3, 5]:
            for lg in [1, 4, 8]:
                for rns in [0, 3, 5]:
                    for ruf in [0, 1]:
                        cfg = make_config(
                            num_warps=nw,
                            num_stages=ns,
                            l2_groupings=[lg],
                            range_num_stages=[0, rns],
                            range_unroll_factors=[0, ruf],
                        )
                        label = f"combo_nw{nw}_ns{ns}_lg{lg}_rns{rns}_ruf{ruf}"
                        r = test_config(cfg, "tf32", k, w, u, g, label=label)
                        results.append(r)
                        combined_count += 1
                        if r[1]:
                            print(f"  {label}: {r[1]*1e6:.1f} us", flush=True)
                        if combined_count % 20 == 0:
                            print(f"  ... tested {combined_count} combos so far", flush=True)

    # =========================================================================
    # 15. COMBINED with different block_sizes and dot_precision
    # =========================================================================
    print("\n=== COMBINED SWEEP 2 (block_sizes + best combos) ===", flush=True)
    for bs in [8, 16, 32, 64]:
        for nw in [4, 8]:
            for ns in [3, 5]:
                for dp in ["ieee", "tf32"]:
                    cfg = make_config(
                        block_sizes=[bs],
                        num_warps=nw,
                        num_stages=ns,
                    )
                    label = f"bs{bs}_nw{nw}_ns{ns}_{dp}"
                    r = test_config(cfg, dp, k, w, u, g, label=label)
                    results.append(r)
                    if r[1]:
                        print(f"  {label}: {r[1]*1e6:.1f} us", flush=True)

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print("\n" + "=" * 80, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 80, flush=True)

    working = [(label, t, cfg, dp) for label, t, cfg, dp in results if t is not None]
    failed = [(label, t, cfg, dp) for label, t, cfg, dp in results if t is None]

    print(f"\nTotal configs tested: {len(results)}", flush=True)
    print(f"Working: {len(working)}", flush=True)
    print(f"Failed: {len(failed)}", flush=True)

    working.sort(key=lambda x: x[1])

    print(f"\n{'Rank':<6} {'Time (us)':<12} {'Label':<60} {'dot_prec':<8}", flush=True)
    print("-" * 90, flush=True)
    for i, (label, t, cfg, dp) in enumerate(working[:30]):
        print(f"{i+1:<6} {t*1e6:<12.1f} {label:<60} {dp:<8}", flush=True)

    if working:
        print(f"\n{'='*80}", flush=True)
        print("TOP 10 FULL CONFIG DETAILS:", flush=True)
        print(f"{'='*80}", flush=True)
        for i, (label, t, cfg, dp) in enumerate(working[:10]):
            print(f"\n--- #{i+1}: {label} ({t*1e6:.1f} us, dot_precision={dp}) ---", flush=True)
            print(f"  block_sizes={cfg.block_sizes}", flush=True)
            print(f"  num_warps={cfg.num_warps}", flush=True)
            print(f"  num_stages={cfg.num_stages}", flush=True)
            print(f"  l2_groupings={cfg.l2_groupings}", flush=True)
            print(f"  loop_orders={cfg.loop_orders}", flush=True)
            print(f"  indexing={cfg.indexing}", flush=True)
            print(f"  load_eviction_policies={cfg.load_eviction_policies}", flush=True)
            print(f"  range_num_stages={cfg.range_num_stages}", flush=True)
            print(f"  range_unroll_factors={cfg.range_unroll_factors}", flush=True)
            print(f"  range_flattens={cfg.range_flattens}", flush=True)
            print(f"  range_multi_buffers={cfg.range_multi_buffers}", flush=True)
            print(f"  pid_type={cfg.pid_type}", flush=True)

    if failed:
        print(f"\n{'='*80}", flush=True)
        print(f"FAILED CONFIGS ({len(failed)}):", flush=True)
        for label, _, _, dp in failed[:20]:
            print(f"  {label} (dot_precision={dp})", flush=True)
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more", flush=True)

    print(f"\n{'='*80}", flush=True)
    print("SWEEP COMPLETE", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
