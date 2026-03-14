#!/usr/bin/env python3
"""
Per-shape autotuning for all 5 kernels using Helion's built-in autotuner.
Creates kernels WITHOUT configs and lets HELION_AUTOTUNE_EFFORT find the best.
Run with: HELION_AUTOTUNE_EFFORT=full python3 autotune_pershape.py
"""
import os
import sys
import time
import torch
import helion
import helion.language as hl

# Ensure full autotuning
if "HELION_AUTOTUNE_EFFORT" not in os.environ:
    os.environ["HELION_AUTOTUNE_EFFORT"] = "full"

def benchmark(fn, args, warmup=20, iters=200):
    """Benchmark a kernel function."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.time() - start) / iters * 1000

def get_best_config(kernel, args):
    """Extract the best config from an autotuned kernel."""
    bound = kernel.bind(args)
    if hasattr(bound, '_config'):
        return repr(bound._config)
    # Try to get config from the kernel's cache
    for key, val in kernel._cache.items():
        if hasattr(val, '_config'):
            return repr(val._config)
    return "unknown"

###############################################################################
# FP8 QUANT
###############################################################################
def autotune_fp8():
    print("=" * 60)
    print("FP8 QUANT - Per-shape autotuning")
    print("=" * 60)

    shapes = [
        # (num_tokens, hidden_dim, group_size)
        (1, 256, 64),
        (4, 512, 128),
        (16, 1024, 64),
        (1, 4096, 128),
        (8, 4096, 128),
        (256, 4096, 128),
        (256, 8192, 128),
        (4096, 7168, 128),
    ]

    for T, H, gsz in shapes:
        N = T * (H // gsz)

        @helion.kernel(static_shapes=True)
        def fp8_kernel(data: torch.Tensor, scales_out: torch.Tensor) -> torch.Tensor:
            nrows = data.size(0)
            ncols = hl.specialize(data.size(1))
            MAX_VAL = 448.0
            qout = torch.empty(nrows, ncols, dtype=torch.float32, device=data.device)
            for rr in hl.tile(nrows):
                row = data[rr, :].to(torch.float32)
                amax = torch.amax(torch.abs(row), -1)
                amax = torch.clamp(amax, min=1e-10)
                scale = amax / MAX_VAL
                qout[rr, :] = row / scale[:, None]
                scales_out[rr] = scale
            return qout

        data = torch.randn(N, gsz, device="cuda", dtype=torch.bfloat16)
        scales = torch.empty(N, device="cuda", dtype=torch.float32)

        try:
            result = fp8_kernel(data, scales)  # Triggers autotuning
            ms = benchmark(fp8_kernel, (data, scales))
            cfg = get_best_config(fp8_kernel, (data, scales))
            print(f"\n({T}, {H}, {gsz}): {ms:.4f}ms")
            print(f"  Config: {cfg}")
        except Exception as e:
            print(f"\n({T}, {H}, {gsz}): FAILED - {e}")

###############################################################################
# CAUSAL CONV1D
###############################################################################
def autotune_conv1d():
    print("\n" + "=" * 60)
    print("CAUSAL CONV1D - Per-shape autotuning")
    print("=" * 60)

    shapes = [
        # (B, D, S, W)
        (1, 64, 64, 4),
        (2, 128, 128, 4),
        (1, 256, 256, 3),
        (1, 128, 64, 8),
        (4, 64, 128, 4),
        (1, 768, 512, 4),
        (1, 768, 2048, 4),
        (1, 1536, 2048, 4),
        (1, 2560, 2048, 4),
        (1, 2560, 4096, 4),
    ]

    for B, D, S, W in shapes:
        @helion.kernel(static_shapes=True)
        def conv1d_kernel(x_pad: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            Bs = x_pad.size(0)
            Ds = x_pad.size(1)
            L = x_pad.size(2)
            Ws = hl.specialize(w.size(1))
            N = L - Ws + 1
            y = torch.empty(Bs, Ds, N, dtype=x_pad.dtype, device=x_pad.device)
            for rb, rd, rs in hl.tile([Bs, Ds, N], block_size=[1, None, None]):
                bi = rb.begin
                acc = hl.zeros([rd, rs], dtype=torch.float32)
                for j in range(Ws):
                    coeff = w[rd, j].to(torch.float32)
                    x_val = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)
                    acc = acc + x_val * coeff[:, None]
                acc = acc + b[rd].to(torch.float32)[:, None]
                y[rb, rd, rs] = acc[None, :, :].to(y.dtype)
            return y

        x_pad = torch.randn(B, D, S + W - 1, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(D, W, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(D, device="cuda", dtype=torch.bfloat16)

        try:
            result = conv1d_kernel(x_pad, w, b)
            ms = benchmark(conv1d_kernel, (x_pad, w, b))
            cfg = get_best_config(conv1d_kernel, (x_pad, w, b))
            print(f"\n({B}, {D}, {S}, {W}): {ms:.4f}ms")
            print(f"  Config: {cfg}")
        except Exception as e:
            print(f"\n({B}, {D}, {S}, {W}): FAILED - {e}")

###############################################################################
# CHUNK FWD H
###############################################################################
def autotune_fwd_h():
    print("\n" + "=" * 60)
    print("CHUNK FWD H - Per-shape autotuning")
    print("=" * 60)

    shapes = [
        (1, 64, 2, 64, 64),
        (2, 128, 4, 64, 64),
        (1, 256, 4, 64, 128),
        (1, 64, 1, 64, 64),
        (2, 512, 3, 64, 64),
        (2, 1024, 3, 64, 64),
        (3, 1024, 4, 100, 100),
        (4, 1024, 4, 128, 128),
        (2, 1536, 4, 128, 128),
        (4, 2048, 8, 64, 64),
    ]

    for B, T, H, K, V in shapes:
        @helion.kernel(static_shapes=True, dot_precision="ieee")
        def fwd_h_kernel(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            Bs, Ts, Hs, Ks = k.shape
            Vs = u.shape[-1]
            C = 64
            Ks = hl.specialize(Ks)
            Vs = hl.specialize(Vs)
            NT = (Ts + C - 1) // C
            h_out = torch.empty(Bs, NT, Hs, Ks, Vs, dtype=k.dtype, device=k.device)
            v_out = torch.empty_like(u)
            BH = Bs * Hs
            for flat, tv in hl.tile([BH, Vs], block_size=[1, 8]):
                b_idx = flat.begin // Hs
                h_idx = flat.begin % Hs
                state = hl.zeros([Ks, tv], dtype=torch.float32)
                for tc in hl.tile(Ts, block_size=C):
                    chunk_idx = tc.begin // C
                    t_end = min(tc.begin + C, Ts) - 1
                    h_out[b_idx, chunk_idx, h_idx, :, tv] = state.to(k.dtype)
                    proj = hl.dot(w[b_idx, tc, h_idx, :], state, out_dtype=torch.float32)
                    diff = u[b_idx, tc, h_idx, tv].to(torch.float32) - proj
                    v_out[b_idx, tc, h_idx, tv] = diff.to(u.dtype)
                    g_end = g[b_idx, t_end, h_idx].to(torch.float32)
                    g_t = g[b_idx, tc, h_idx].to(torch.float32)
                    valid = tc.index < Ts
                    alpha = torch.where(valid, torch.exp(g_end - g_t), 0.0)
                    k_adj = k[b_idx, tc, h_idx, :] * alpha[:, None]
                    state = state * torch.exp(g_end)
                    upd = hl.dot(k_adj.T, diff, out_dtype=torch.float32)
                    state = state + upd
            return h_out, v_out

        k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
        u = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)

        try:
            result = fwd_h_kernel(k, w, u, g)
            ms = benchmark(fwd_h_kernel, (k, w, u, g))
            cfg = get_best_config(fwd_h_kernel, (k, w, u, g))
            print(f"\n({B}, {T}, {H}, {K}, {V}): {ms:.4f}ms")
            print(f"  Config: {cfg}")
        except Exception as e:
            print(f"\n({B}, {T}, {H}, {K}, {V}): FAILED - {e}")

###############################################################################
# CHUNK FWD O
###############################################################################
def autotune_fwd_o():
    print("\n" + "=" * 60)
    print("CHUNK FWD O - Per-shape autotuning")
    print("=" * 60)

    shapes = [
        (1, 64, 2, 64, 64),
        (2, 128, 4, 64, 64),
        (1, 256, 4, 64, 128),
        (1, 64, 1, 64, 64),
        (2, 512, 3, 64, 64),
        (2, 1024, 3, 64, 64),
        (3, 1024, 4, 100, 100),
        (4, 1024, 4, 128, 128),
        (2, 1536, 4, 128, 128),
        (4, 2048, 8, 64, 64),
    ]

    for B, T, H, K, V in shapes:
        @helion.kernel(static_shapes=True, dot_precision="ieee")
        def fwd_o_kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h: torch.Tensor, g: torch.Tensor, scale: float) -> torch.Tensor:
            Bs, Ts, Hs, Ks = q.shape
            Vs = v.shape[-1]
            C = 64
            Ks = hl.specialize(Ks)
            Vs = hl.specialize(Vs)
            out = torch.empty_like(v)
            BH = Bs * Hs
            for flat_bh, tile_t in hl.tile([BH, Ts], block_size=[1, C]):
                b_idx = flat_bh.begin // Hs
                h_idx = flat_bh.begin % Hs
                c_idx = tile_t.begin // C
                g_vals = g[b_idx, tile_t, h_idx].to(torch.float32)
                q_chunk = q[b_idx, tile_t, h_idx, :].to(torch.float32)
                k_chunk = k[b_idx, tile_t, h_idx, :].to(torch.float32)
                v_chunk = v[b_idx, tile_t, h_idx, :]
                qk = hl.dot(q_chunk, k_chunk.T)
                g_diff = g_vals[:, None] - g_vals[None, :]
                qk = qk * torch.exp(g_diff)
                idx = hl.arange(tile_t.block_size)
                mask = idx[:, None] >= idx[None, :]
                qk = torch.where(mask, qk, 0.0)
                local_out = hl.dot(qk.to(v.dtype), v_chunk)
                q_g = q_chunk * torch.exp(g_vals)[:, None]
                global_out = hl.dot(q_g, h[b_idx, c_idx, h_idx, :, :])
                out[b_idx, tile_t, h_idx, :] = ((global_out + local_out) * scale).to(out.dtype)
            return out

        q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
        NT = (T + 63) // 64
        h = torch.randn(B, NT, H, K, V, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
        scale = K ** -0.5

        try:
            result = fwd_o_kernel(q, k, v, h, g, scale)
            ms = benchmark(fwd_o_kernel, (q, k, v, h, g, scale))
            cfg = get_best_config(fwd_o_kernel, (q, k, v, h, g, scale))
            print(f"\n({B}, {T}, {H}, {K}, {V}): {ms:.4f}ms")
            print(f"  Config: {cfg}")
        except Exception as e:
            print(f"\n({B}, {T}, {H}, {K}, {V}): FAILED - {e}")

###############################################################################
# RECOMPUTE W/U
###############################################################################
def autotune_recompute():
    print("\n" + "=" * 60)
    print("RECOMPUTE W/U - Per-shape autotuning")
    print("=" * 60)

    shapes = [
        (1, 64, 2, 64, 64),
        (2, 128, 4, 64, 64),
        (1, 256, 4, 64, 128),
        (1, 64, 1, 64, 64),
        (2, 512, 3, 64, 64),
        (2, 1024, 3, 64, 64),
        (3, 1024, 4, 100, 100),
        (4, 1024, 4, 128, 128),
        (2, 1536, 4, 128, 128),
        (4, 2048, 8, 64, 64),
    ]

    for B, T, H, K, V in shapes:
        @helion.kernel(static_shapes=True, dot_precision="ieee")
        def recompute_kernel(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            Bs, Ts, Hs, Ks = k.shape
            Vs = v.shape[-1]
            C = hl.specialize(A.shape[-1])
            Ks = hl.specialize(Ks)
            Vs = hl.specialize(Vs)
            w_out = torch.empty_like(k)
            u_out = torch.empty_like(v)
            BH = Bs * Hs
            for flat_bh, rt in hl.tile([BH, Ts], block_size=[1, C]):
                b_idx = flat_bh.begin // Hs
                h_idx = flat_bh.begin % Hs
                beta_vals = beta[b_idx, rt, h_idx].to(torch.float32)
                g_vals = g[b_idx, rt, h_idx].to(torch.float32)
                k_chunk = k[b_idx, rt, h_idx, :].to(torch.float32)
                v_chunk = v[b_idx, rt, h_idx, :].to(torch.float32)
                A_chunk = A[b_idx, rt, h_idx, :].to(torch.float32)
                k_scaled = k_chunk * (beta_vals * torch.exp(g_vals))[:, None]
                v_scaled = v_chunk * beta_vals[:, None]
                w_result = hl.dot(A_chunk, k_scaled, out_dtype=torch.float32)
                u_result = hl.dot(A_chunk, v_scaled, out_dtype=torch.float32)
                w_out[b_idx, rt, h_idx, :] = w_result.to(k.dtype)
                u_out[b_idx, rt, h_idx, :] = u_result.to(v.dtype)
            return w_out, u_out

        k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(B, T, H, 64, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16)

        try:
            result = recompute_kernel(k, v, beta, A, g)
            ms = benchmark(recompute_kernel, (k, v, beta, A, g))
            cfg = get_best_config(recompute_kernel, (k, v, beta, A, g))
            print(f"\n({B}, {T}, {H}, {K}, {V}): {ms:.4f}ms")
            print(f"  Config: {cfg}")
        except Exception as e:
            print(f"\n({B}, {T}, {H}, {K}, {V}): FAILED - {e}")

###############################################################################
# MAIN - Run specific kernel or all
###############################################################################
if __name__ == "__main__":
    kernel = sys.argv[1] if len(sys.argv) > 1 else "all"

    if kernel in ("fp8", "all"):
        autotune_fp8()
    if kernel in ("conv1d", "all"):
        autotune_conv1d()
    if kernel in ("fwd_h", "all"):
        autotune_fwd_h()
    if kernel in ("fwd_o", "all"):
        autotune_fwd_o()
    if kernel in ("recompute", "all"):
        autotune_recompute()

    print("\n\n=== ALL AUTOTUNING COMPLETE ===")
