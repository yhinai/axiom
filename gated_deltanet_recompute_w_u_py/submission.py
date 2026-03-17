#!POPCORN leaderboard gated_deltanet_recompute_w_u
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=1),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=1),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=2),
    # Benchmark shapes
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=3),
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=3),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=3),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=3),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=3),
}

def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
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

        acc_dtype = torch.float32
        w_out = torch.empty_like(k)
        u_out = torch.empty_like(v)
        block_k = hl.register_block_size(K)
        block_v = hl.register_block_size(V)

        for tile_b, tile_t, tile_h in hl.tile([B, T, H], block_size=[1, C, 1]):
            b_idx = tile_b.id
            h_idx = tile_h.id
            a_chunk = A[b_idx, tile_t, h_idx, :].to(acc_dtype)
            beta_vals = beta[b_idx, tile_t, h_idx].to(acc_dtype)[:, None]
            gate_vals = beta_vals * torch.exp(g[b_idx, tile_t, h_idx].to(acc_dtype))[:, None]

            for tile_k in hl.tile(K, block_size=block_k):
                w_out[b_idx, tile_t, h_idx, tile_k] = hl.dot(
                    a_chunk,
                    k[b_idx, tile_t, h_idx, tile_k].to(acc_dtype) * gate_vals,
                    out_dtype=acc_dtype,
                ).to(k.dtype)

            for tile_v in hl.tile(V, block_size=block_v):
                u_out[b_idx, tile_t, h_idx, tile_v] = hl.dot(
                    a_chunk,
                    v[b_idx, tile_t, h_idx, tile_v].to(acc_dtype) * beta_vals,
                    out_dtype=acc_dtype,
                ).to(v.dtype)

        return w_out, u_out

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    return _KERNELS[(B, T, H, K, V)](k, v, beta, A, g)
