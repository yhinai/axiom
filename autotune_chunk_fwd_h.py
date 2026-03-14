#!/usr/bin/env python3
"""
Run HELION_AUTOTUNE_EFFORT=full on the chunk_fwd_h kernel to find Helion's best config.
"""
import os
os.environ["HELION_AUTOTUNE_EFFORT"] = "full"

import torch
import helion
import helion.language as hl
import time

_LOG2E = 1.4426950408889634

@helion.kernel(static_shapes=True, dot_precision="tf32")
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
    print(f"HELION_AUTOTUNE_EFFORT={os.environ.get('HELION_AUTOTUNE_EFFORT', 'not set')}", flush=True)
    print("=" * 80, flush=True)
    print("Running autotuning (this will take a while)...", flush=True)

    # First call triggers autotuning
    t0 = time.time()
    result = kernel(k, w, u, g)
    torch.cuda.synchronize()
    autotune_time = time.time() - t0
    print(f"\nAutotuning completed in {autotune_time:.1f}s", flush=True)

    # Print the best config found
    print("\n" + "=" * 80, flush=True)
    print("BEST CONFIG FOUND BY AUTOTUNE:", flush=True)
    print("=" * 80, flush=True)

    # Access the autotuned config
    try:
        best_config = kernel.config
        print(f"Config: {best_config}", flush=True)
        print(f"\nDetailed fields:", flush=True)
        print(f"  block_sizes={best_config.block_sizes}", flush=True)
        print(f"  num_warps={best_config.num_warps}", flush=True)
        print(f"  num_stages={best_config.num_stages}", flush=True)
        print(f"  l2_groupings={best_config.l2_groupings}", flush=True)
        print(f"  loop_orders={best_config.loop_orders}", flush=True)
        print(f"  indexing={best_config.indexing}", flush=True)
        print(f"  load_eviction_policies={best_config.load_eviction_policies}", flush=True)
        print(f"  range_num_stages={best_config.range_num_stages}", flush=True)
        print(f"  range_unroll_factors={best_config.range_unroll_factors}", flush=True)
        print(f"  range_flattens={best_config.range_flattens}", flush=True)
        print(f"  range_multi_buffers={best_config.range_multi_buffers}", flush=True)
        print(f"  pid_type={best_config.pid_type}", flush=True)
    except Exception as e:
        print(f"Could not access config: {e}", flush=True)

    # Benchmark the autotuned kernel
    print("\nBenchmarking autotuned kernel...", flush=True)
    warmup = 10
    iters = 50
    for _ in range(warmup):
        kernel(k, w, u, g)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        kernel(k, w, u, g)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times.sort()
    median = times[len(times) // 2]
    best = times[0]
    print(f"Median time: {median*1e6:.1f} us", flush=True)
    print(f"Best time: {best*1e6:.1f} us", flush=True)
    print(f"Mean time: {sum(times)/len(times)*1e6:.1f} us", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("AUTOTUNE COMPLETE", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
