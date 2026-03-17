#!POPCORN leaderboard gated_deltanet_chunk_fwd_o
#!POPCORN gpu B200_Nebius

import os

os.environ["ENABLE_TILE"] = "1"
os.environ["HELION_BACKEND"] = "tileir"

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# VG2: TileIR + per-shape LFBO-autotuned configs from B200.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes (B, T, H, K, V)
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[], num_ctas=1, occupancy=4, num_stages=4, indexing="tensor_descriptor"),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[], num_ctas=1, occupancy=4, num_stages=4, indexing="tensor_descriptor"),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[], num_ctas=1, occupancy=4, num_stages=4, indexing="tensor_descriptor"),
    # Benchmark shapes — best from LFBO autotuning log
    (1, 64, 1, 64, 64): helion.Config(
        block_sizes=[],
        indexing=['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'],
        l2_groupings=[1], load_eviction_policies=['', '', '', '', ''],
        loop_orders=[[1, 0]], maxnreg=128, num_ctas=1, num_sm_multiplier=2,
        num_stages=6, num_warps=4, occupancy=2, pid_type='persistent_interleaved',
        range_warp_specializes=[],
    ),
    (2, 512, 3, 64, 64): helion.Config(
        block_sizes=[],
        indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'],
        l2_groupings=[1], load_eviction_policies=['', '', '', '', ''],
        loop_orders=[[1, 0]], maxnreg=256, num_ctas=1, num_sm_multiplier=2,
        num_stages=8, num_warps=4, occupancy=1, pid_type='persistent_blocked',
        range_warp_specializes=[],
    ),
    (2, 1024, 3, 64, 64): helion.Config(
        block_sizes=[],
        indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'],
        l2_groupings=[64], load_eviction_policies=['', '', '', '', ''],
        loop_orders=[[0, 1]], num_ctas=1, num_sm_multiplier=1,
        num_stages=8, num_warps=4, occupancy=1, pid_type='persistent_interleaved',
        range_warp_specializes=[],
    ),
    # Extra benchmark shapes — use safe defaults
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[], num_ctas=1, occupancy=4, num_stages=4, indexing="tensor_descriptor"),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[], num_ctas=1, occupancy=4, num_stages=4, indexing="tensor_descriptor"),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[], num_ctas=1, occupancy=4, num_stages=4, indexing="tensor_descriptor"),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[], num_ctas=1, occupancy=4, num_stages=4, indexing="tensor_descriptor"),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        q: torch.Tensor,      # [B, T, H, K] fp32
        k: torch.Tensor,      # [B, T, H, K] fp32
        v: torch.Tensor,      # [B, T, H, V] fp32
        h: torch.Tensor,      # [B, NT, H, K, V] fp32
        g: torch.Tensor,      # [B, T, H] fp32
        scale: float,
    ) -> torch.Tensor:
        B, T, H, K = q.shape
        V = v.shape[-1]
        C = 64
        NT = T // C
        K = hl.specialize(K)
        V = hl.specialize(V)

        out = torch.empty(B, T, H, V, dtype=torch.float32, device=q.device)

        BH = B * H
        for flat_bh, flat_c in hl.tile([BH, NT], block_size=[1, 1]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            c_idx = flat_c.begin
            t0 = c_idx * C

            # Load chunk data
            g_vals = g[b_idx, t0:t0 + C, h_idx]
            q_tile = q[b_idx, t0:t0 + C, h_idx, :]
            k_tile = k[b_idx, t0:t0 + C, h_idx, :]
            v_tile = v[b_idx, t0:t0 + C, h_idx, :]
            h_tile = h[b_idx, c_idx, h_idx, :, :]

            # Intra-chunk: causal attention
            qk = hl.dot(q_tile, k_tile.T)

            idx_range = hl.arange(C)
            g_diff = g_vals[:, None] - g_vals[None, :]
            causal_mask = idx_range[:, None] >= idx_range[None, :]
            sim = torch.where(causal_mask, qk * torch.exp(g_diff), 0.0)

            local_out = hl.dot(sim, v_tile)

            # Inter-chunk: query saved hidden state
            q_scaled = q_tile * torch.exp(g_vals)[:, None]
            global_out = hl.dot(q_scaled, h_tile)

            out[b_idx, t0:t0 + C, h_idx, :] = (global_out + local_out) * scale

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
