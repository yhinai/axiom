from task import input_t, output_t

import torch
import helion
import helion.language as hl


# B200-tuned configs caused 12min timeout on leaderboard; use single safe config.
OPTIMIZED = helion.Config(block_sizes=[], num_warps=4, num_stages=2)
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
#     helion.Config(..., advanced_controls_file="/opt/booster_pack/chunk_fwd_h_0.acf")


# NOTE: This is an intentionally inefficient baseline implementation.
def _make_kernel(config: helion.Config | None):
    if config is None:
        decorator = helion.kernel(static_shapes=True, dot_precision="ieee", autotune_effort="quick")
    else:
        decorator = helion.kernel(static_shapes=True, dot_precision="ieee", config=config)

    @decorator
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

        for flat, tv in hl.tile([BH, V], block_size=[1, 8]):
            b_idx = flat.begin // H
            h_idx = flat.begin % H
            state = hl.zeros([K, tv], dtype=torch.float32)

            for tc in hl.tile(T, block_size=C):
                chunk_idx = tc.begin // C
                t_end = min(tc.begin + C, T) - 1

                h_out[b_idx, chunk_idx, h_idx, :, tv] = state.to(k.dtype)

                proj = hl.dot(
                    w[b_idx, tc, h_idx, :], state, out_dtype=torch.float32
                )
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


def get_autotune_kernel():
    return _make_kernel(None)


def data_to_kernel_args(data: input_t):
    return data  # (k, w, u, g)


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(k, w, u, g)
