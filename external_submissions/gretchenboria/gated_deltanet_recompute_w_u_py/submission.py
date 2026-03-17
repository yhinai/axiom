#!POPCORN leaderboard gated_deltanet_recompute_w_u
#!POPCORN gpu B200_Nebius

from task import input_t, output_t
import torch
import helion
import helion.language as hl
import glob

# Optimization Strategy:
# 1. Eliminated the redundant double-pass loops.
# 2. Vectorized the matrix multiplication: 
#    Instead of doing element-wise accumulation across C, we treat A as a [C, C] tile
#    and k/v as [C, K]/[C, V] matrices. We scale k and v directly and use hl.dot().

acf_files = glob.glob("/opt/booster_pack/recompute_w_u_fwd_*.acf")

config_params = {
    "block_sizes": [], # Using default C=64 block size
    "num_warps": [2, 4, 8],
    "num_stages": [2, 3, 4]
}

if acf_files:
    config_params["advanced_controls_file"] = acf_files

TUNE_CONFIG = helion.Config(**config_params)

# --- HARDCODED LEADERBOARD CONFIGS ---
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[], num_warps=2, num_stages=2),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    # Benchmark shapes
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[], num_warps=2, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=2),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[], num_warps=8, num_stages=2),
}

DEFAULT_CONFIG = helion.Config(block_sizes=[], num_warps=4, num_stages=2)

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

            # Load A tile [C, C]
            A_tile = A[b_idx, rt, h_idx, :].to(torch.float32)

            # Load scaling vectors
            beta_tile = beta[b_idx, rt, h_idx].to(torch.float32)
            g_tile = g[b_idx, rt, h_idx].to(torch.float32)
            decay_tile = torch.exp(g_tile)

            # Load data chunks
            k_tile = k[b_idx, rt, h_idx, :].to(torch.float32)
            v_tile = v[b_idx, rt, h_idx, :].to(torch.float32)

            # Apply element-wise scaling
            k_scaled = k_tile * (beta_tile * decay_tile)[:, None]
            v_scaled = v_tile * beta_tile[:, None]

            # Vectorized matrix multiplication: A @ scaled_vectors
            w_acc = hl.dot(A_tile, k_scaled)
            u_acc = hl.dot(A_tile, v_scaled)

            w_out[b_idx, rt, h_idx, :] = w_acc.to(k.dtype)
            u_out[b_idx, rt, h_idx, :] = u_acc.to(v.dtype)

        return w_out, u_out

    return kernel

_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}
_DEFAULT_KERNEL = _make_kernel(DEFAULT_CONFIG)

def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    kernel = _KERNELS.get((B, T, H, K, V), _DEFAULT_KERNEL)
    return kernel(k, v, beta, A, g)
