from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
# Autotune locally for each shape, then paste the best config here.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[], num_stages=1, num_warps=8, pid_type='flat'),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[], num_stages=1, num_warps=8, pid_type='flat'),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[], num_stages=1, num_warps=8, pid_type='flat'),
    # Benchmark shapes
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[], num_stages=1, num_warps=8, pid_type='flat'),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[], num_stages=1, num_warps=8, pid_type='flat'),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[], num_stages=1, num_warps=8, pid_type='flat'),
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[], num_stages=1, num_warps=16, pid_type='flat'),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[], loop_orders=[[1, 0]], num_stages=1, num_warps=32, pid_type='flat'),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[], loop_orders=[[1, 0]], num_stages=1, num_warps=32, pid_type='flat'),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[], num_stages=1, num_warps=8, pid_type='flat'),
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
        V = hl.specialize(v.shape[-1])
        C = hl.specialize(A.shape[-1])
        K = hl.specialize(K)

        w_out = torch.empty_like(k)
        u_out = torch.empty_like(v)

        BH = B * H
        for flat_bh, rt in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H

            A_chunk = A[b_idx, rt, h_idx, :].to(torch.float32)
            k_chunk = k[b_idx, rt, h_idx, :].to(torch.float32)
            v_chunk = v[b_idx, rt, h_idx, :].to(torch.float32)
            beta_chunk = beta[b_idx, rt, h_idx].to(torch.float32)
            g_chunk = g[b_idx, rt, h_idx].to(torch.float32)

            v_scaled = v_chunk * beta_chunk[:, None]
            k_scaled = k_chunk * (beta_chunk * torch.exp(g_chunk))[:, None]

            u = hl.dot(A_chunk, v_scaled, out_dtype=torch.float32)
            w = hl.dot(A_chunk, k_scaled, out_dtype=torch.float32)

            w_out[b_idx, rt, h_idx, :] = w.to(k.dtype)
            u_out[b_idx, rt, h_idx, :] = u.to(v.dtype)

        return w_out, u_out

    return kernel


_KERNEL_CACHE: dict[tuple, object] = {}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    key = (B, T, H, K, V)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = _make_kernel(SHAPE_CONFIGS[key])
    kernel = _KERNEL_CACHE[key]
    return kernel(k, v, beta, A, g)
