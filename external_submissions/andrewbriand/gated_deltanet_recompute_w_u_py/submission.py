from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
# Autotune locally for each shape, then paste the best config here.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=1),  # TODO: use any config that passes correctness check
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=1),  # TODO: use any config that passes correctness check
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=1),  # TODO: use any config that passes correctness check
    # Benchmark shapes
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=1),  # TODO: replace with your autotuned config
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=1),  # TODO: replace with your autotuned config
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=1),  # TODO: replace with your autotuned config
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[], num_warps=4, num_stages=1),  # TODO: replace with your autotuned config
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=1),  # TODO: replace with your autotuned config
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=1),  # TODO: replace with your autotuned config
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=1),  # TODO: replace with your autotuned config
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
        for flat_bh, rt, rk, rv in hl.tile([BH, T, K, V], block_size=[1, C, 32, 32]):
            A_block = A[flat_bh.begin // H, rt, flat_bh.begin % H, :].to(torch.float32)
            coeff = beta[flat_bh.begin // H, rt, flat_bh.begin % H].to(torch.float32)
            decay = torch.exp(g[flat_bh.begin // H, rt, flat_bh.begin % H].to(torch.float32))

            k_block = k[flat_bh.begin // H, rt, flat_bh.begin % H, rk].to(torch.float32)
            k_scaled = k_block * (coeff * decay)[:, None]
            w_out[flat_bh.begin // H, rt, flat_bh.begin % H, rk] = hl.dot(A_block, k_scaled).to(k.dtype)

            v_block = v[flat_bh.begin // H, rt, flat_bh.begin % H, rv].to(torch.float32)
            v_scaled = v_block * coeff[:, None]
            u_out[flat_bh.begin // H, rt, flat_bh.begin % H, rv] = hl.dot(A_block, v_scaled).to(v.dtype)

        return w_out, u_out
    return kernel

def _make_kernel_old(config: helion.Config):
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

            # A block: [C, C] — rows = rt positions, cols = ci positions
            A_block = A[b_idx, rt, h_idx, :].to(torch.float32)  # [C, C]

            # Scale k by beta * exp(g) along the C (source) dimension
            # Each ci position gets its own scalar: beta[ci] * exp(g[ci])
            #t_range = rt.begin + torch.arange(C, device=k.device)

            coeff = beta[b_idx, rt, h_idx].to(torch.float32)        # [C]
            decay = torch.exp(g[b_idx, rt, h_idx].to(torch.float32)) # [C]

            k_block = k[b_idx, rt, h_idx, :].to(torch.float32)  # [C, K]
            #v_block = v[b_idx, rt, h_idx, :].to(torch.float32)  # [C, V]

            # Apply per-row scaling to k and v before the matmul
            k_scaled = k_block * (coeff * decay)[:, None]  # [C, K]
            #v_scaled = v_block * coeff[:, None]             # [C, V]

            # hl.dot replaces the manual outer-product accumulation loop:
            #   sum_ci  A[rt, ci] * (k_ci * coeff_ci * decay_ci)  →  A_block @ k_scaled
            #   sum_ci  A[rt, ci] * (v_ci * coeff_ci)             →  A_block @ v_scaled
            w_acc1 = hl.dot(A_block, k_scaled)  # [C, K]
            #u_acc1 = hl.dot(A_block, v_scaled)  # [C, V]

            w_out[b_idx, rt, h_idx, :] = w_acc1.to(k.dtype)
            #u_out[b_idx, rt, h_idx, :] = u_acc1.to(v.dtype)
        return w_out, u_out
    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(k, v, beta, A, g)
