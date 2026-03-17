#!POPCORN leaderboard gated_deltanet_recompute_w_u
#!POPCORN gpu B200_Nebius

import os

os.environ["ENABLE_TILE"] = "1"
os.environ["HELION_BACKEND"] = "tileir"

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# VG1: TileIR + per-shape configs + no dot_precision="ieee" (tf32 tensor cores)
# Fully parallel — no sequential accumulation, so tf32 precision is safe.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes (B, T, H, K, V)
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[], num_ctas=1, occupancy=4, num_stages=4, indexing="tensor_descriptor"),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[], num_ctas=1, occupancy=4, num_stages=4, indexing="tensor_descriptor"),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[], num_ctas=1, occupancy=4, num_stages=4, indexing="tensor_descriptor"),
    # Benchmark shapes
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[], num_ctas=1, occupancy=4, num_stages=4, indexing="tensor_descriptor"),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[], num_ctas=1, occupancy=8, num_stages=6, indexing="tensor_descriptor"),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[], num_ctas=2, occupancy=8, num_stages=6, indexing="tensor_descriptor"),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        k: torch.Tensor,     # [B, T, H, K] fp32
        v: torch.Tensor,     # [B, T, H, V] fp32
        beta: torch.Tensor,  # [B, T, H] fp32
        A: torch.Tensor,     # [B, T, H, C] fp32
        g: torch.Tensor,     # [B, T, H] fp32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = v.shape[-1]
        C = hl.specialize(A.shape[-1])
        NT = T // C
        K = hl.specialize(K)
        V = hl.specialize(V)

        w_out = torch.empty_like(k)
        u_out = torch.empty_like(v)

        BH = B * H
        for flat_bh, flat_c in hl.tile([BH, NT], block_size=[1, 1]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            c_idx = flat_c.begin
            t0 = c_idx * C

            k_tile = k[b_idx, t0:t0 + C, h_idx, :]
            v_tile = v[b_idx, t0:t0 + C, h_idx, :]
            beta_vals = beta[b_idx, t0:t0 + C, h_idx]
            A_tile = A[b_idx, t0:t0 + C, h_idx, :]
            g_vals = g[b_idx, t0:t0 + C, h_idx]

            # u = A @ (v * beta[:, None])
            v_scaled = v_tile * beta_vals[:, None]
            u_chunk = hl.dot(A_tile, v_scaled)

            # w = A @ (k * (beta * exp(g))[:, None])
            k_scaled = k_tile * (beta_vals * torch.exp(g_vals))[:, None]
            w_chunk = hl.dot(A_tile, k_scaled)

            w_out[b_idx, t0:t0 + C, h_idx, :] = w_chunk.to(w_out.dtype)
            u_out[b_idx, t0:t0 + C, h_idx, :] = u_chunk.to(u_out.dtype)

        return w_out, u_out

    return kernel


_KERNELS: dict[tuple, object] = {}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    key = (B, T, H, K, V)

    if key not in _KERNELS:
        _KERNELS[key] = _make_kernel(SHAPE_CONFIGS[key])

    return _KERNELS[key](k, v, beta, A, g)
