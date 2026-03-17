from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
# Autotune locally for each shape, then paste the best config here.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes — V=64: full V tile; V=128: half tile to fit registers
    (1, 64, 2, 64, 64): helion.Config(
        block_sizes=[64], num_warps=4, num_stages=3,
        indexing="block_ptr",
    ),
    (2, 128, 4, 64, 64): helion.Config(
        block_sizes=[64], num_warps=4, num_stages=3,
        indexing="block_ptr",
    ),
    (1, 256, 4, 64, 128): helion.Config(
        block_sizes=[64], num_warps=4, num_stages=3,
        indexing="block_ptr",
    ),
    # Benchmark shapes
    # (1, 64, 1, 64, 64): tiny shape, only 1 chunk — minimize overhead
    (1, 64, 1, 64, 64): helion.Config(
        block_sizes=[32], num_warps=2, num_stages=3,
        indexing="block_ptr",
    ),

    # EDIT THESE
    # (2, 512, 3, 64, 64): 8 chunks, BH=6 — pipeline chunk loop
    (2, 512, 3, 64, 64): helion.Config(
        block_sizes=[32], num_warps=2, num_stages=3,
        range_num_stages=[3], range_multi_buffers=[True],
        indexing="block_ptr", l2_groupings=[4],
    ),
    # (2, 1024, 3, 64, 64): 16 chunks, BH=6 — longer recurrence benefits from pipelining
    (2, 1024, 3, 64, 64): helion.Config(
        block_sizes=[32], num_warps=2, num_stages=3,
        range_num_stages=[3], range_multi_buffers=[True],
        indexing="block_ptr", l2_groupings=[4],
    ),
    # (3, 1024, 4, 100, 100): non-power-of-2 K/V, BH=12
    (3, 1024, 4, 100, 100): helion.Config(
        block_sizes=[64], num_warps=4, num_stages=3,
        range_num_stages=[3], range_multi_buffers=[True],
        indexing="block_ptr", l2_groupings=[4],
    ),
    # (4, 1024, 4, 128, 128): large K/V=128, BH=16 — more warps for bigger matmuls
    (4, 1024, 4, 128, 128): helion.Config(
        block_sizes=[64], num_warps=8, num_stages=4,
        range_num_stages=[4], range_multi_buffers=[True],
        indexing="block_ptr", l2_groupings=[4],
    ),
    # (2, 1536, 4, 128, 128): long sequence + large K/V
    (2, 1536, 4, 128, 128): helion.Config(
        block_sizes=[64], num_warps=8, num_stages=4,
        range_num_stages=[4], range_multi_buffers=[True],
        indexing="block_ptr", l2_groupings=[4],
    ),
    # (4, 2048, 8, 64, 64): 32 chunks, BH=32 — lots of parallelism
    (4, 2048, 8, 64, 64): helion.Config(
        block_sizes=[64], num_warps=4, num_stages=3,
        range_num_stages=[3], range_multi_buffers=[True],
        indexing="block_ptr", l2_groupings=[8],
    ),
}


# Optional: add advanced_controls_file to your Config for extra performance (see docs).
# Autotune with autotune_search_acf to find the best ACF, then hardcode it:
#     helion.Config(..., advanced_controls_file="/opt/booster_pack/chunk_fwd_h_0.acf")


# NOTE: This is an intentionally inefficient baseline implementation.
def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(
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
                alpha = torch.where(valid, torch.exp(g_end - g_t), 0.0)
                k_adj = k[b_idx, tc, h_idx, :] * alpha[:, None]

                state = state * torch.exp(g_end)
                upd = hl.dot(k_adj.T, diff, out_dtype=torch.float32)
                state = state + upd

        return h_out, v_out

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(k, w, u, g)
