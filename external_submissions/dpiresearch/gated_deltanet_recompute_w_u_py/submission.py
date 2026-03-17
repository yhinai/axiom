from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Single config for reliability on leaderboard (12min timeout).
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
#     helion.Config(..., advanced_controls_file="/opt/booster_pack/recompute_w_u_fwd_0.acf")


# NOTE: This is an intentionally inefficient baseline implementation.
def _make_kernel(config: helion.Config | None):
    if config is None:
        decorator = helion.kernel(static_shapes=True, dot_precision="ieee", autotune_effort="quick")
    else:
        decorator = helion.kernel(static_shapes=True, dot_precision="ieee", config=config)

    @decorator
    def kernel(
        k: torch.Tensor,     # [B, T, H, K]
        v: torch.Tensor,     # [B, T, H, V]
        beta: torch.Tensor,  # [B, T, H]
        A: torch.Tensor,     # [B, T, H, BT]
        g: torch.Tensor,     # [B, T, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

            w_acc1 = hl.zeros([rt, K], dtype=torch.float32)
            u_acc1 = hl.zeros([rt, V], dtype=torch.float32)
            w_acc2 = hl.zeros([rt, K], dtype=torch.float32)
            u_acc2 = hl.zeros([rt, V], dtype=torch.float32)

            for ci in range(C):
                t_ci = rt.begin + ci
                a_col = A[b_idx, rt, h_idx, ci].to(torch.float32)
                coeff_ci = beta[b_idx, t_ci, h_idx].to(torch.float32)
                decay_ci = torch.exp(g[b_idx, t_ci, h_idx].to(torch.float32))

                k_ci = k[b_idx, t_ci, h_idx, :].to(torch.float32)
                v_ci = v[b_idx, t_ci, h_idx, :].to(torch.float32)

                w_acc1 = w_acc1 + a_col[:, None] * (k_ci * coeff_ci * decay_ci)[None, :]
                u_acc1 = u_acc1 + a_col[:, None] * (v_ci * coeff_ci)[None, :]

            for ci in range(C - 1, -1, -1):
                t_ci = rt.begin + ci
                a_col = A[b_idx, rt, h_idx, ci].to(torch.float32)
                coeff_ci = beta[b_idx, t_ci, h_idx].to(torch.float32)
                decay_ci = torch.exp(g[b_idx, t_ci, h_idx].to(torch.float32))

                k_ci = k[b_idx, t_ci, h_idx, :].to(torch.float32)
                v_ci = v[b_idx, t_ci, h_idx, :].to(torch.float32)

                w_acc2 = w_acc2 + a_col[:, None] * (k_ci * coeff_ci * decay_ci)[None, :]
                u_acc2 = u_acc2 + a_col[:, None] * (v_ci * coeff_ci)[None, :]

            w_out[b_idx, rt, h_idx, :] = ((w_acc1 + w_acc2) * 0.5).to(k.dtype)
            u_out[b_idx, rt, h_idx, :] = ((u_acc1 + u_acc2) * 0.5).to(v.dtype)

        return w_out, u_out

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def get_autotune_kernel():
    return _make_kernel(None)


def data_to_kernel_args(data: input_t):
    return data  # (k, v, beta, A, g)


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(k, v, beta, A, g)
