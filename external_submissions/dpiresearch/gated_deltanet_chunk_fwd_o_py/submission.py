from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Single config for reliability on leaderboard (12min timeout).
OPTIMIZED = helion.Config(block_sizes=[], num_warps=8, num_stages=4)
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    (1, 64, 2, 64, 64): OPTIMIZED,
    (2, 128, 4, 64, 64): OPTIMIZED,
    (1, 256, 4, 64, 128): OPTIMIZED,
    (1, 64, 1, 64, 64): OPTIMIZED,
    (2, 512, 3, 64, 64): OPTIMIZED,
    (2, 1024, 3, 64, 64): OPTIMIZED,
    (3, 1024, 4, 100, 100): OPTIMIZED,
    (4, 1024, 4, 128, 128): OPTIMIZED,
    (2, 1536, 4, 128, 128): OPTIMIZED,
    (4, 2048, 8, 64, 64): OPTIMIZED,
}


# Optional: add advanced_controls_file to your Config for extra performance (see docs).
# Autotune with autotune_search_acf to find the best ACF, then hardcode it:
#     helion.Config(..., advanced_controls_file="/opt/booster_pack/chunk_fwd_o_0.acf")


# NOTE: This is an intentionally inefficient baseline implementation.
def _make_kernel(config: helion.Config | None):
    if config is None:
        decorator = helion.kernel(static_shapes=True, dot_precision="ieee", autotune_effort="quick")
    else:
        decorator = helion.kernel(static_shapes=True, dot_precision="ieee", config=config)

    @decorator
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


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def get_autotune_kernel():
    return _make_kernel(None)


def data_to_kernel_args(data: input_t):
    q, k, v_new, h, g = data
    K = q.shape[-1]
    return (q, k, v_new, h, g, K ** -0.5)


def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    scale = K ** -0.5
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(q, k, v_new, h, g, scale)
