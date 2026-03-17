#!POPCORN leaderboard gated_deltanet_chunk_fwd_o
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Single ACF-autotuned config for all shapes (avoids per-shape compilation overhead)
_TUNED = helion.Config(advanced_controls_file='/opt/booster_pack/chunk_fwd_o_0.acf', block_sizes=[], indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', '', '', '', ''], loop_orders=[[1, 0]], num_stages=1, num_warps=16, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None])

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): _TUNED,
    (2, 128, 4, 64, 64): _TUNED,
    (1, 256, 4, 64, 128): _TUNED,
    (1, 64, 1, 128, 128): _TUNED,
    (2, 128, 2, 100, 100): _TUNED,
    # Benchmark shapes
    (1, 64, 1, 64, 64): _TUNED,
    (2, 512, 3, 64, 64): _TUNED,
    (2, 1024, 3, 64, 64): _TUNED,
    (3, 1024, 4, 100, 100): _TUNED,
    (4, 1024, 4, 128, 128): _TUNED,
    (2, 1536, 4, 128, 128): _TUNED,
    (4, 2048, 8, 64, 64): _TUNED,
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
        g: torch.Tensor,
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
            q_chunk = q[b_idx, tile_t, h_idx, :].to(torch.float32)
            k_chunk = k[b_idx, tile_t, h_idx, :].to(torch.float32)
            v_chunk = v[b_idx, tile_t, h_idx, :]

            # exp2 is a single GPU instruction vs multi-instruction exp
            LOG2E = 1.4426950408889634

            # Intra-chunk: qk with gating and causal mask
            qk = hl.dot(q_chunk, k_chunk.T)
            g_diff = g_vals[:, None] - g_vals[None, :]
            qk = qk * torch.exp2(g_diff * LOG2E)
            idx = hl.arange(tile_t.block_size)
            mask = idx[:, None] >= idx[None, :]
            qk = torch.where(mask, qk, 0.0)
            # Compute local_out first, then accumulate global_out into it via acc=
            acc = hl.dot(qk.to(v.dtype), v_chunk)

            # Inter-chunk: (q * exp(g)) @ h, fused add via acc=
            q_g = q_chunk * torch.exp2(g_vals * LOG2E)[:, None]
            acc = hl.dot(q_g, h[b_idx, c_idx, h_idx, :, :], acc=acc)

            out[b_idx, tile_t, h_idx, :] = (acc * scale).to(out.dtype)

        return out

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    scale = K ** -0.5
    shape = (B, T, H, K, V)
    if shape not in _KERNELS:
        _KERNELS[shape] = _make_kernel(_TUNED)
    kernel = _KERNELS[shape]
    return kernel(q, k, v_new, h, g, scale)
