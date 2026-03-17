from task import input_t, output_t

import torch
import helion
import helion.language as hl


_DEFAULT_CONFIG = helion.Config(block_sizes=[], num_warps=8, num_stages=2)


# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
# Autotune locally for each shape, then paste the best config here.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): _DEFAULT_CONFIG,
    (2, 128, 4, 64, 64): _DEFAULT_CONFIG,
    (1, 256, 4, 64, 128): _DEFAULT_CONFIG,
    # Benchmark shapes
    (1, 64, 1, 64, 64): _DEFAULT_CONFIG,
    (2, 512, 3, 64, 64): _DEFAULT_CONFIG,
    (2, 1024, 3, 64, 64): _DEFAULT_CONFIG,
    (3, 1024, 4, 100, 100): _DEFAULT_CONFIG,
    (4, 1024, 4, 128, 128): _DEFAULT_CONFIG,
    (2, 1536, 4, 128, 128): _DEFAULT_CONFIG,
    (4, 2048, 8, 64, 64): _DEFAULT_CONFIG,
}


# Optional: add advanced_controls_file to your Config for extra performance (see docs).
# Autotune with autotune_search_acf to find the best ACF, then hardcode it:
#     helion.Config(..., advanced_controls_file="/opt/booster_pack/recompute_w_u_fwd_0.acf")

def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
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

            a_tile = A[b_idx, rt, h_idx, :].to(torch.float32)
            beta_tile = beta[b_idx, rt, h_idx].to(torch.float32)
            g_tile = g[b_idx, rt, h_idx].to(torch.float32)

            v_scaled = v[b_idx, rt, h_idx, :].to(torch.float32) * beta_tile[:, None]
            w_scaled = k[b_idx, rt, h_idx, :].to(torch.float32) * (
                beta_tile * torch.exp(g_tile)
            )[:, None]

            w_out[b_idx, rt, h_idx, :] = hl.dot(a_tile, w_scaled, out_dtype=torch.float32).to(k.dtype)
            u_out[b_idx, rt, h_idx, :] = hl.dot(a_tile, v_scaled, out_dtype=torch.float32).to(v.dtype)

        return w_out, u_out

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(k, v, beta, A, g)
