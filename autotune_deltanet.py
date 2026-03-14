#!/usr/bin/env python3
"""
Per-shape autotuning for gated_deltanet kernels (fwd_h, fwd_o, recompute_w_u).
Uses DESurrogateHybrid autotuner and tries multiple strategies.
"""
import os
import sys
import time
import copy
import torch
import helion
import helion.language as hl

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "full")

def benchmark(fn, args, warmup=20, iters=200):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.time() - start) / iters * 1000

###############################################################################
# Config variations to try for each kernel
###############################################################################
def gen_fwd_h_configs():
    """Generate config variations for chunk_fwd_h."""
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
    configs = [helion.Config(**base)]

    # Vary num_stages
    for ns in [1, 2, 4, 5, 7]:
        c = {**base, 'num_stages': ns}
        configs.append(helion.Config(**c))

    # Vary num_warps
    for nw in [2, 8, 16]:
        c = {**base, 'num_warps': nw}
        configs.append(helion.Config(**c))

    # Try persistent PID (like recompute_w_u's 2.4x win)
    for pid in ['persistent_blocked', 'persistent_interleaved']:
        for nsm in [4, 16, 64]:
            for mreg in [32, 64, 128]:
                c = {**base, 'pid_type': pid, 'num_sm_multiplier': nsm, 'maxnreg': mreg}
                configs.append(helion.Config(**c))

    # Vary range_num_stages
    for rns in [0, 1, 2, 4]:
        c = {**base, 'range_num_stages': [0, rns]}
        configs.append(helion.Config(**c))

    # Vary range_unroll_factors
    for ruf in [1, 2, 4]:
        c = {**base, 'range_unroll_factors': [0, ruf]}
        configs.append(helion.Config(**c))

    # Warp specialization
    c = {**base, 'range_warp_specializes': [True, None]}
    configs.append(helion.Config(**c))
    c = {**base, 'range_warp_specializes': [None, True]}
    configs.append(helion.Config(**c))

    # Different loop orders
    c = {**base, 'loop_orders': [[0, 1]]}
    configs.append(helion.Config(**c))

    return configs

def gen_fwd_o_configs():
    """Generate config variations for chunk_fwd_o."""
    base = dict(
        block_sizes=[],
        indexing=['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'],
        pid_type='flat',
        num_warps=16,
        num_stages=1,
        l2_groupings=[1],
        loop_orders=[[1, 0]],
        load_eviction_policies=['', '', '', '', ''],
        range_flattens=[None],
        range_multi_buffers=[None],
        range_num_stages=[0],
        range_unroll_factors=[0],
        range_warp_specializes=[None],
    )
    configs = [helion.Config(**base)]

    # Vary num_warps
    for nw in [4, 8, 32]:
        c = {**base, 'num_warps': nw}
        configs.append(helion.Config(**c))

    # Vary num_stages
    for ns in [2, 3, 5]:
        c = {**base, 'num_stages': ns}
        configs.append(helion.Config(**c))

    # Try tensor_descriptor for more loads
    idx_combos = [
        ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'],
        ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'],
        ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'],
    ]
    for idx in idx_combos:
        c = {**base, 'indexing': idx}
        configs.append(helion.Config(**c))

    # Persistent PID
    for pid in ['persistent_blocked', 'persistent_interleaved']:
        for nsm in [4, 16]:
            for mreg in [32, 64]:
                c = {**base, 'pid_type': pid, 'num_sm_multiplier': nsm, 'maxnreg': mreg, 'num_warps': 16}
                configs.append(helion.Config(**c))

    # Eviction policies
    for ep in [['first', '', '', '', ''], ['', 'first', '', '', ''], ['first', 'first', '', '', '']]:
        c = {**base, 'load_eviction_policies': ep}
        configs.append(helion.Config(**c))

    # Warp specialization
    c = {**base, 'range_warp_specializes': [True]}
    configs.append(helion.Config(**c))

    return configs

def gen_recompute_configs():
    """Generate config variations for recompute_w_u."""
    base = dict(
        block_sizes=[],
        indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'],
        pid_type='persistent_blocked',
        num_warps=32,
        num_stages=1,
        l2_groupings=[16],
        loop_orders=[[0, 1]],
        load_eviction_policies=['', 'first', '', 'first', ''],
        maxnreg=32,
        num_sm_multiplier=16,
        range_flattens=[None],
        range_multi_buffers=[False],
        range_num_stages=[3],
        range_unroll_factors=[4],
        range_warp_specializes=[None],
    )
    configs = [helion.Config(**base)]

    # Vary num_sm_multiplier
    for nsm in [4, 8, 32, 64, 128]:
        c = {**base, 'num_sm_multiplier': nsm}
        configs.append(helion.Config(**c))

    # Vary maxnreg
    for mreg in [64, 128, 256]:
        c = {**base, 'maxnreg': mreg}
        configs.append(helion.Config(**c))

    # Vary range_unroll_factors
    for ruf in [0, 1, 2]:
        c = {**base, 'range_unroll_factors': [ruf]}
        configs.append(helion.Config(**c))

    # Vary range_num_stages
    for rns in [0, 1, 2, 4]:
        c = {**base, 'range_num_stages': [rns]}
        configs.append(helion.Config(**c))

    # Different num_warps
    for nw in [4, 8, 16]:
        c = {**base, 'num_warps': nw}
        configs.append(helion.Config(**c))

    # Interleaved PID
    c = {**base, 'pid_type': 'persistent_interleaved'}
    configs.append(helion.Config(**c))

    # Different l2_groupings
    for lg in [1, 4, 8, 32, 64]:
        c = {**base, 'l2_groupings': [lg]}
        configs.append(helion.Config(**c))

    # Warp specialization
    c = {**base, 'range_warp_specializes': [True]}
    configs.append(helion.Config(**c))

    return configs

###############################################################################
# Kernel functions
###############################################################################
def make_fwd_h(config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

def make_fwd_o(config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h: torch.Tensor, g: torch.Tensor, scale: float) -> torch.Tensor:
        B, T, H, K = q.shape
        V = v.shape[-1]
        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)
        out = torch.empty_like(v)
        BH = B * H
        for flat_bh, tile_t in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
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
    return kernel

def make_recompute(config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, A: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = v.shape[-1]
        C = hl.specialize(A.shape[-1])
        K = hl.specialize(K)
        V = hl.specialize(V)
        w_out = torch.empty_like(k)
        u_out = torch.empty_like(v)
        BH = B * H
        for flat_bh, rt in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
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
    return kernel

###############################################################################
# Main tuning loop
###############################################################################
def tune_kernel(kernel_name, make_fn, configs, shapes, make_args_fn):
    print(f"\n{'='*60}")
    print(f"{kernel_name} - {len(configs)} configs x {len(shapes)} shapes")
    print(f"{'='*60}")

    best_per_shape = {}

    for shape in shapes:
        print(f"\n--- Shape: {shape} ---")
        best_ms = float('inf')
        best_cfg = None
        best_cfg_repr = None
        args = make_args_fn(shape)

        for i, config in enumerate(configs):
            try:
                kernel = make_fn(config)
                result = kernel(*args)
                ms = benchmark(kernel, args, warmup=10, iters=100)

                if ms < best_ms:
                    best_ms = ms
                    best_cfg = config
                    best_cfg_repr = repr(config)

                if i % 10 == 0 or ms < best_ms * 1.01:
                    print(f"  [{i}/{len(configs)}] {ms:.4f}ms {'*** BEST' if ms <= best_ms else ''}")
            except Exception as e:
                err = str(e)[:80]
                if i < 3:
                    print(f"  [{i}/{len(configs)}] FAILED: {err}")

        best_per_shape[shape] = (best_ms, best_cfg_repr)
        print(f"  BEST: {best_ms:.4f}ms")
        print(f"  Config: {best_cfg_repr}")

    print(f"\n{'='*60}")
    print(f"{kernel_name} - SUMMARY")
    print(f"{'='*60}")
    for shape, (ms, cfg) in best_per_shape.items():
        print(f"  {shape}: {ms:.4f}ms")
        print(f"    {cfg}")

    return best_per_shape

if __name__ == "__main__":
    kernel = sys.argv[1] if len(sys.argv) > 1 else "all"

    deltanet_shapes = [
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

    if kernel in ("fwd_h", "all"):
        def make_fwd_h_args(shape):
            B, T, H, K, V = shape
            return (
                torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16),
                torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16),
                torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16),
                torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16),
            )
        tune_kernel("CHUNK_FWD_H", make_fwd_h, gen_fwd_h_configs(), deltanet_shapes, make_fwd_h_args)

    if kernel in ("fwd_o", "all"):
        def make_fwd_o_args(shape):
            B, T, H, K, V = shape
            NT = (T + 63) // 64
            return (
                torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16),
                torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16),
                torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16),
                torch.randn(B, NT, H, K, V, device="cuda", dtype=torch.bfloat16),
                torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16),
                K ** -0.5,
            )
        tune_kernel("CHUNK_FWD_O", make_fwd_o, gen_fwd_o_configs(), deltanet_shapes, make_fwd_o_args)

    if kernel in ("recompute", "all"):
        def make_recompute_args(shape):
            B, T, H, K, V = shape
            return (
                torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16),
                torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16),
                torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16),
                torch.randn(B, T, H, 64, device="cuda", dtype=torch.bfloat16),
                torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16),
            )
        tune_kernel("RECOMPUTE_W_U", make_recompute, gen_recompute_configs(), deltanet_shapes, make_recompute_args)

    print("\n\n=== DELTANET AUTOTUNING COMPLETE ===")
