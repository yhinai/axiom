#!POPCORN leaderboard gated_deltanet_chunk_fwd_h
#!POPCORN gpu B200_Nebius
from __future__ import annotations

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Benchmark shapes only — test shapes use default config
    (1, 64, 1, 64, 64): helion.Config(advanced_controls_file='/opt/booster_pack/chunk_fwd_h_0.acf', block_sizes=[8], indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', '', 'first', '', ''], loop_orders=[[0, 1]], num_stages=1, num_warps=16, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 0], range_unroll_factors=[0, 0], range_warp_specializes=[None, None], static_ranges=[False]),
    (2, 512, 3, 64, 64): helion.Config(advanced_controls_file='/opt/booster_pack/chunk_fwd_h_0.acf', block_sizes=[8], indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', '', '', '', ''], loop_orders=[[0, 1]], num_stages=1, num_warps=4, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 3], range_unroll_factors=[0, 0], range_warp_specializes=[None, None], static_ranges=[False]),
    (2, 1024, 3, 64, 64): helion.Config(advanced_controls_file='/opt/booster_pack/chunk_fwd_h_0.acf', block_sizes=[8], indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', '', '', '', ''], loop_orders=[[0, 1]], num_stages=1, num_warps=2, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 3], range_unroll_factors=[0, 1], range_warp_specializes=[None, None]),
}


@helion.kernel(
    static_shapes=True,
    autotune_effort="none",
)
def gdn_chunk_fwd_h_kernel(
    k: torch.Tensor,   # [B, T, H, K]
    w: torch.Tensor,   # [B, T, H, K]
    u: torch.Tensor,   # [B, T, H, V]
    g: torch.Tensor,   # [B, T, H]
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    V = u.shape[-1]
    C = 64
    K = hl.specialize(K)
    V = hl.specialize(V)
    NT = T // C
    BH = B * H

    h_out = torch.empty(B, NT, H, K, V, dtype=k.dtype, device=k.device)
    v_out = torch.empty_like(u)

    bv = hl.register_block_size(V)
    for flat, tv in hl.tile([BH, V], block_size=[1, bv]):
        b_idx = flat.begin // H
        h_idx = flat.begin % H
        state = hl.zeros([K, tv], dtype=torch.float32)

        for tc in hl.tile(T, block_size=C):
            chunk_idx = tc.begin // C
            t_end = tc.begin + C - 1

            h_out[b_idx, chunk_idx, h_idx, :, tv] = state

            proj = hl.dot(w[b_idx, tc, h_idx, :], state, out_dtype=torch.float32)
            v_new = u[b_idx, tc, h_idx, tv] - proj
            v_out[b_idx, tc, h_idx, tv] = v_new

            g_last = g[b_idx, t_end, h_idx]
            alpha = torch.exp(g_last - g[b_idx, tc, h_idx])

            k_adj = k[b_idx, tc, h_idx, :] * alpha[:, None]
            state = state * torch.exp(g_last)
            state = hl.dot(k_adj.T, v_new, acc=state, out_dtype=torch.float32)

    return h_out, v_out


# Pre-compile and warm up a runner for each shape
_RUNNERS: dict[tuple, object] = {}
for (_B, _T, _H, _K, _V), _cfg in SHAPE_CONFIGS.items():
    _ek = torch.empty(_B, _T, _H, _K, dtype=torch.float32, device="cuda")
    _ew = torch.empty(_B, _T, _H, _K, dtype=torch.float32, device="cuda")
    _eu = torch.empty(_B, _T, _H, _V, dtype=torch.float32, device="cuda")
    _eg = torch.empty(_B, _T, _H, dtype=torch.float32, device="cuda")
    _bound = gdn_chunk_fwd_h_kernel.bind((_ek, _ew, _eu, _eg))
    _runner = _bound.compile_config(_cfg)
    _runner(_ek, _ew, _eu, _eg)  # warmup
    _RUNNERS[(_B, _T, _H, _K, _V)] = _runner
torch.cuda.synchronize()


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    key = (B, T, H, K, V)
    if key in _RUNNERS:
        return _RUNNERS[key](k, w, u, g)
    return gdn_chunk_fwd_h_kernel(k, w, u, g)
