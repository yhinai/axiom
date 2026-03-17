#!POPCORN leaderboard gated_deltanet_recompute_w_u
#!POPCORN gpu B200_Nebius
from __future__ import annotations

import torch
import helion
import helion.language as hl
from task import input_t, output_t

CHUNK_SIZE = 64

# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    (1, 64, 1, 64, 64): helion.Config(advanced_controls_file='/opt/booster_pack/recompute_w_u_fwd_0.acf', block_sizes=[8], indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', '', '', '', ''], loop_orders=[[0, 1, 2]], num_sm_multiplier=1, num_stages=2, num_warps=2, pid_type='persistent_blocked', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None]),
    (2, 512, 3, 64, 64): helion.Config(advanced_controls_file='/opt/booster_pack/recompute_w_u_fwd_2.acf', block_sizes=[32], indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', '', '', '', ''], loop_orders=[[0, 1, 2]], num_stages=1, num_warps=8, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None]),
    (2, 1024, 3, 64, 64): helion.Config(advanced_controls_file='/opt/booster_pack/recompute_w_u_fwd_2.acf', block_sizes=[64], indexing=['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], l2_groupings=[4], load_eviction_policies=['', '', '', '', ''], loop_orders=[[2, 0, 1]], num_stages=1, num_warps=4, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None]),
}


@helion.kernel(autotune_effort="none")
def recompute_w_u_kernel(
    k: torch.Tensor,       # [B, T, H, K]
    v: torch.Tensor,       # [B, T, H, V]
    beta: torch.Tensor,    # [B, T, H]
    A: torch.Tensor,       # [B, T, H, BT]
    g: torch.Tensor,       # [B, T, H]
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    V = v.shape[-1]
    C = CHUNK_SIZE
    K = hl.specialize(K)
    V = hl.specialize(V)
    BH = B * H

    w_out = torch.empty_like(k)
    u_out = torch.empty_like(v)

    bv = hl.register_block_size(V)

    for flat, tc, tv in hl.tile([BH, T, V], block_size=[1, C, bv]):
        b_idx = flat.begin // H
        h_idx = flat.begin % H
        A_mat = A[b_idx, tc, h_idx, :]              # [C, BT]
        beta_vec = beta[b_idx, tc, h_idx]            # [C]

        # u = A @ diag(beta) @ v — tiled over V
        v_scaled = v[b_idx, tc, h_idx, tv] * beta_vec[:, None]
        u_out[b_idx, tc, h_idx, tv] = hl.dot(A_mat, v_scaled, out_dtype=torch.float32)

        # w = A @ diag(beta * exp(g)) @ k — full K with ':'
        g_vec = g[b_idx, tc, h_idx]
        scale_w = beta_vec * torch.exp(g_vec)
        k_scaled = k[b_idx, tc, h_idx, :] * scale_w[:, None]
        w_out[b_idx, tc, h_idx, :] = hl.dot(A_mat, k_scaled, out_dtype=torch.float32)

    return w_out, u_out


# Pre-compile and warm up a runner for each shape
_RUNNERS: dict[tuple, object] = {}

for (_B, _T, _H, _K, _V), _cfg in SHAPE_CONFIGS.items():
    _ek = torch.empty(_B, _T, _H, _K, dtype=torch.float32, device="cuda")
    _ev = torch.empty(_B, _T, _H, _V, dtype=torch.float32, device="cuda")
    _ebeta = torch.empty(_B, _T, _H, dtype=torch.float32, device="cuda")
    _eA = torch.empty(_B, _T, _H, CHUNK_SIZE, dtype=torch.float32, device="cuda")
    _eg = torch.empty(_B, _T, _H, dtype=torch.float32, device="cuda")

    _bound = recompute_w_u_kernel.bind((_ek, _ev, _ebeta, _eA, _eg))
    _runner = _bound.compile_config(_cfg)
    _runner(_ek, _ev, _ebeta, _eA, _eg)
    _RUNNERS[(_B, _T, _H, _K, _V)] = _runner

if SHAPE_CONFIGS:
    torch.cuda.synchronize()


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    key = (B, T, H, K, V)
    if key in _RUNNERS:
        return _RUNNERS[key](k, v, beta, A, g)
    return recompute_w_u_kernel(k, v, beta, A, g)
