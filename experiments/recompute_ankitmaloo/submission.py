from task import input_t, output_t

import torch
import helion
import helion.language as hl


CHUNK_SIZE = 64


# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
# Known tuned shapes use the best configurations found in the manual rounds.
# Larger extrapolated shapes use the same direct-layout kernel with conservative settings.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=1),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=1),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=2),
    # Benchmarks / tuned leaderboard shapes
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=3),
    # Extrapolated larger shapes
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=3),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=3),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=3),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=3),
}


def _default_config(shape: tuple[int, int, int, int, int]) -> helion.Config:
    _, T, _, K, V = shape
    large_d = max(K, V) > 64
    long_t = T >= 1024
    return helion.Config(
        block_sizes=[64, 64],
        num_warps=8 if T >= 512 else 4,
        num_stages=3 if large_d or long_t else 2,
    )


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
        out_w = torch.empty_like(k)
        out_u = torch.empty_like(v)
        block_k = hl.register_block_size(K)
        block_v = hl.register_block_size(V)

        for tile_b, tile_t, tile_h in hl.tile([B, T, H], block_size=[1, C, 1]):
            i_b = tile_b.id
            i_h = tile_h.id
            b_A = A[i_b, tile_t, i_h, :].to(acc_dtype)
            beta_t = beta[i_b, tile_t, i_h].to(acc_dtype)
            gate_t = (beta_t * torch.exp(g[i_b, tile_t, i_h].to(acc_dtype)))[:, None]
            beta_t = beta_t[:, None]

            for tile_k in hl.tile(K, block_size=block_k):
                out_w[i_b, tile_t, i_h, tile_k] = hl.dot(
                    b_A,
                    k[i_b, tile_t, i_h, tile_k].to(acc_dtype) * gate_t,
                    out_dtype=acc_dtype,
                ).to(k.dtype)

            for tile_v in hl.tile(V, block_size=block_v):
                out_u[i_b, tile_t, i_h, tile_v] = hl.dot(
                    b_A,
                    v[i_b, tile_t, i_h, tile_v].to(acc_dtype) * beta_t,
                    out_dtype=acc_dtype,
                ).to(v.dtype)

        return out_w, out_u

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    shape = (B, T, H, K, V)
    kernel = _KERNELS.get(shape)
    if kernel is None:
        kernel = _make_kernel(_default_config(shape))
        _KERNELS[shape] = kernel
    return kernel(k, v, beta, A, g)
