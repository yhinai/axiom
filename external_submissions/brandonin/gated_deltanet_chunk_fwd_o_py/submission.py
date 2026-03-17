#!POPCORN leaderboard gated_deltanet_chunk_fwd_o
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
# block_sizes=[] because all tile block sizes are specified explicitly
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[], num_warps=8, num_stages=2, advanced_controls_file="/opt/booster_pack/chunk_fwd_o_0.acf"),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[], num_warps=8, num_stages=2, advanced_controls_file="/opt/booster_pack/chunk_fwd_o_0.acf"),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[], num_warps=8, num_stages=2, advanced_controls_file="/opt/booster_pack/chunk_fwd_o_0.acf"),
    # Benchmark shapes — full-effort autotuned configs (tf32x3, ACF sweep)
    (1, 64, 1, 64, 64): helion.Config(
        advanced_controls_file='/opt/booster_pack/chunk_fwd_o_3.acf',
        block_sizes=[],
        indexing=['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'],
        l2_groupings=[1], load_eviction_policies=['', '', 'first', 'last', 'first'],
        loop_orders=[[0, 1]], num_stages=1, num_warps=8, pid_type='flat',
        range_flattens=[None], range_multi_buffers=[None],
        range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None],
    ),
    (2, 512, 3, 64, 64): helion.Config(
        advanced_controls_file='/opt/booster_pack/chunk_fwd_o_6.acf',
        block_sizes=[],
        indexing=['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'],
        l2_groupings=[1], load_eviction_policies=['', 'last', 'last', '', ''],
        loop_orders=[[0, 1]], num_stages=1, num_warps=8, pid_type='flat',
        range_flattens=[None], range_multi_buffers=[None],
        range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None],
    ),
    (2, 1024, 3, 64, 64): helion.Config(
        advanced_controls_file='/opt/booster_pack/chunk_fwd_o_0.acf',
        block_sizes=[],
        indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'],
        l2_groupings=[1], load_eviction_policies=['', '', '', 'last', ''],
        loop_orders=[[0, 1]], num_stages=1, num_warps=8, pid_type='flat',
        range_flattens=[None], range_multi_buffers=[None],
        range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None],
    ),
    # Extra shapes (fallback)
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[], num_warps=8, num_stages=2, advanced_controls_file="/opt/booster_pack/chunk_fwd_o_0.acf"),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[], num_warps=8, num_stages=2, advanced_controls_file="/opt/booster_pack/chunk_fwd_o_0.acf"),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[], num_warps=8, num_stages=2, advanced_controls_file="/opt/booster_pack/chunk_fwd_o_0.acf"),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[], num_warps=8, num_stages=2, advanced_controls_file="/opt/booster_pack/chunk_fwd_o_0.acf"),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="tf32x3", fast_math=True, config=config)
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

            g_vals = g[b_idx, tile_t, h_idx].to(torch.float32)
            q_vals = q[b_idx, tile_t, h_idx, :]
            k_vals = k[b_idx, tile_t, h_idx, :]
            v_vals = v[b_idx, tile_t, h_idx, :]

            # Inter-chunk: (q @ h) * exp(g) — exp(g) is small (g very negative), safe
            global_out = hl.dot(q_vals, h[b_idx, c_idx, h_idx, :, :], out_dtype=torch.float32)
            global_out = global_out * torch.exp(g_vals)[:, None]

            # Intra-chunk: causal_mask(q @ k^T * exp(g_i - g_j)) @ v
            # Compute q @ k^T first (no scaling by exp(g))
            sim = hl.dot(q_vals, k_vals.T, out_dtype=torch.float32)
            # Apply gating: exp(g_i - g_j) — bounded within chunk, numerically safe
            g_diff = g_vals[:, None] - g_vals[None, :]
            sim = sim * torch.exp(g_diff)
            # Causal mask
            idx = hl.arange(tile_t.block_size)
            mask = idx[:, None] >= idx[None, :]
            sim = torch.where(mask, sim, 0.0)
            local_out = hl.dot(sim.to(v.dtype), v_vals, out_dtype=torch.float32)

            out[b_idx, tile_t, h_idx, :] = ((global_out + local_out) * scale).to(out.dtype)

        return out

    return kernel


_KERNELS: dict = {}


def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    scale = K ** -0.5
    shape = (B, T, H, K, V)
    if shape not in _KERNELS:
        _KERNELS[shape] = _make_kernel(SHAPE_CONFIGS[shape])
    return _KERNELS[shape](q, k, v_new, h, g, scale)
