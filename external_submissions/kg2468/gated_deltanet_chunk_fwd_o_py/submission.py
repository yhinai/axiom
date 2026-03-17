#!POPCORN leaderboard gated_deltanet_chunk_fwd_o
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes (autotuned)
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[], indexing=['block_ptr', 'block_ptr', 'block_ptr', 'pointer', 'block_ptr', 'pointer'], l2_groupings=[4], load_eviction_policies=['', '', 'first', 'last', ''], loop_orders=[[0, 1]], num_stages=5, num_warps=8, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[]),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[], indexing=['pointer', 'pointer', 'block_ptr', 'block_ptr', 'pointer', 'block_ptr'], l2_groupings=[1], load_eviction_policies=['last', 'first', 'last', '', 'first'], loop_orders=[[0, 1]], num_stages=1, num_warps=16, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[]),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[], indexing=['block_ptr', 'pointer', 'block_ptr', 'pointer', 'pointer', 'block_ptr'], l2_groupings=[16], load_eviction_policies=['last', '', '', '', 'last'], loop_orders=[[0, 1]], num_sm_multiplier=32, num_stages=3, num_warps=16, pid_type='persistent_blocked', range_flattens=[False], range_multi_buffers=[None], range_num_stages=[1], range_unroll_factors=[2], range_warp_specializes=[]),
    # Benchmark shapes (partially autotuned)
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[], indexing=['pointer', 'pointer', 'block_ptr', 'block_ptr', 'pointer', 'block_ptr'], l2_groupings=[16], load_eviction_policies=['last', 'first', 'first', '', ''], loop_orders=[[1, 0]], num_stages=4, num_warps=16, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[]),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[], indexing=['pointer', 'block_ptr', 'pointer', 'block_ptr', 'block_ptr', 'block_ptr'], l2_groupings=[8], load_eviction_policies=['last', 'last', '', 'first', 'first'], loop_orders=[[0, 1]], maxnreg=128, num_sm_multiplier=2, num_stages=8, num_warps=16, pid_type='persistent_interleaved', range_flattens=[None], range_multi_buffers=[False], range_num_stages=[1], range_unroll_factors=[0], range_warp_specializes=[]),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[], num_stages=2, num_warps=8),
    # Ranked shapes (not autotuned - using best defaults)
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[], num_warps=8, num_stages=3),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[], num_warps=8, num_stages=3),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[], num_warps=8, num_stages=3),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[], num_warps=8, num_stages=3),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(
        q: torch.Tensor,     # [B, T, H, K]
        k: torch.Tensor,     # [B, T, H, K]
        v: torch.Tensor,     # [B, T, H, V]
        h: torch.Tensor,     # [B, NT, H, K, V]
        g: torch.Tensor,     # [B, T, H]
        scale: float,
    ) -> torch.Tensor:
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

            g_vals = g[b_idx, tile_t, h_idx]
            q_tile = q[b_idx, tile_t, h_idx, :]
            k_tile = k[b_idx, tile_t, h_idx, :]
            v_tile = v[b_idx, tile_t, h_idx, :]

            # intra-chunk: q @ k^T * exp(g_i - g_j), with causal mask
            qk = hl.dot(q_tile, k_tile.T)
            idx = hl.arange(tile_t.block_size)
            g_diff = g_vals[:, None] - g_vals[None, :]
            causal_mask = idx[:, None] >= idx[None, :]
            sim = torch.where(causal_mask, qk * torch.exp(g_diff), 0.0)
            local_out = hl.dot(sim.to(v.dtype), v_tile)

            # inter-chunk: (q @ h) * exp(g)
            q_s = q_tile * torch.exp(g_vals)[:, None]
            global_out = hl.dot(q_s, h[b_idx, c_idx, h_idx, :, :])

            out[b_idx, tile_t, h_idx, :] = ((global_out + local_out) * scale).to(out.dtype)

        return out

    return kernel


_KERNELS: dict[tuple, object] = {}


def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    scale = K ** -0.5
    key = (B, T, H, K, V)
    if key not in _KERNELS:
        _KERNELS[key] = _make_kernel(SHAPE_CONFIGS[key])
    return _KERNELS[key](q, k, v_new, h, g, scale)
