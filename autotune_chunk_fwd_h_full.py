#!/usr/bin/env python3
"""
Run HELION_AUTOTUNE_EFFORT=full on the chunk_fwd_h kernel.
Uses HELION_AUTOTUNE_TIMEOUT to prevent infinite hangs.
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


torch.manual_seed(2146)
k = torch.randn(2, 1024, 3, 64, device="cuda") / 64**0.5
w = torch.randn(2, 1024, 3, 64, device="cuda")
u = torch.randn(2, 1024, 3, 64, device="cuda")
g = torch.randn(2, 1024, 3, device="cuda")

print(f"Shape: (2, 1024, 3, 64, 64)", flush=True)
print(f"Device: {torch.cuda.get_device_name()}", flush=True)
print(f"HELION_AUTOTUNE_EFFORT=full", flush=True)
print("=" * 80, flush=True)
print("Running full autotuning...", flush=True)

t0 = time.time()
result = kernel(k, w, u, g)
torch.cuda.synchronize()
print(f"\nAutotune took {time.time()-t0:.1f}s", flush=True)

# Benchmark
for _ in range(10):
    kernel(k, w, u, g)
torch.cuda.synchronize()
times = []
for _ in range(50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    kernel(k, w, u, g)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
times.sort()
print(f"\nMedian time: {times[len(times)//2]*1e6:.1f} us", flush=True)
print(f"Best time: {times[0]*1e6:.1f} us", flush=True)
print(f"Mean time: {sum(times)/len(times)*1e6:.1f} us", flush=True)

print("\n" + "=" * 80, flush=True)
print("AUTOTUNE COMPLETE", flush=True)
print("=" * 80, flush=True)
