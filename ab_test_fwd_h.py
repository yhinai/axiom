#!/usr/bin/env python3
"""A/B test config variants for chunk_fwd_h on all 3 benchmark shapes."""
import torch, helion, helion.language as hl, time, math

_LOG2E = 1.4426950408889634

def make_and_bench(cfg, k, w, u, g, label, iters=30):
    @helion.kernel(static_shapes=True, dot_precision="tf32", config=cfg)
    def kern(k, w, u, g):
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
    for _ in range(10):
        kern(k, w, u, g)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        kern(k, w, u, g)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    med = times[iters // 2] * 1000
    mn = times[0] * 1000
    print(f"  {label}: median={med:.4f}ms min={mn:.4f}ms")
    return med

base = dict(
    block_sizes=[],
    indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'],
    l2_groupings=[1],
    load_eviction_policies=['first', '', '', '', ''],
    loop_orders=[[1, 0]],
    num_stages=3,
    num_warps=4,
    pid_type='flat',
    range_flattens=[None, True],
    range_multi_buffers=[None, False],
    range_num_stages=[0, 3],
    range_unroll_factors=[0, 0],
    range_warp_specializes=[None, None],
)

configs = {
    "current": helion.Config(**base),
    "nw=2": helion.Config(**{**base, 'num_warps': 2}),
    "nw=8": helion.Config(**{**base, 'num_warps': 8}),
    "nw=16": helion.Config(**{**base, 'num_warps': 16}),
    "l2g=4": helion.Config(**{**base, 'l2_groupings': [4]}),
    "l2g=8": helion.Config(**{**base, 'l2_groupings': [8]}),
    "ns=5": helion.Config(**{**base, 'num_stages': 5}),
    "rns=[0,0]": helion.Config(**{**base, 'range_num_stages': [0, 0]}),
    "ruf=[0,1]": helion.Config(**{**base, 'range_unroll_factors': [0, 1]}),
    "lo=[0,1]": helion.Config(**{**base, 'loop_orders': [[0, 1]]}),
    # Combined best from sweep
    "sweep_best": helion.Config(**{**base, 'l2_groupings': [8], 'range_num_stages': [0, 0], 'range_unroll_factors': [0, 1]}),
}

shapes = [(1, 64, 1, 64, 64), (2, 512, 3, 64, 64), (2, 1024, 3, 64, 64)]
results = {}

for name, cfg in configs.items():
    print(f"Config: {name}")
    shape_times = []
    for B, T, H, K, V in shapes:
        k = torch.randn(B, T, H, K, device="cuda") / K**0.5
        w = torch.randn(B, T, H, K, device="cuda")
        u = torch.randn(B, T, H, V, device="cuda")
        g = torch.randn(B, T, H, device="cuda")
        try:
            ms = make_and_bench(cfg, k, w, u, g, f"({B},{T},{H})")
            shape_times.append(ms)
        except Exception as e:
            print(f"  ({B},{T},{H}): FAIL {str(e)[:60]}")
            shape_times.append(999)
    geo = math.exp(sum(math.log(t) for t in shape_times) / len(shape_times))
    results[name] = (geo, shape_times)
    print(f"  geomean={geo:.4f}ms")
    print()

print("=" * 70)
print("RANKING BY GEOMEAN:")
print("=" * 70)
print(f"{'Rank':<5} {'Config':<15} {'Geomean':>10} {'Shape0':>10} {'Shape1':>10} {'Shape2':>10}")
print("-" * 70)
for rank, (name, (geo, times)) in enumerate(sorted(results.items(), key=lambda x: x[1][0]), 1):
    cur = " <--" if name == "current" else ""
    print(f"  #{rank:<3} {name:<15} {geo:>8.4f}ms  {times[0]:>8.4f}  {times[1]:>8.4f}  {times[2]:>8.4f}{cur}")
