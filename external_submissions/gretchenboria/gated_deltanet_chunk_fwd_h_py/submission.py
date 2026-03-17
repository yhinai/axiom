#!POPCORN leaderboard gated_deltanet_chunk_fwd_h
#!POPCORN gpu B200_Nebius

from task import input_t, output_t
import torch
import helion
import helion.language as hl
import glob

# Optimization Strategy:
# 1. Eliminate redundant hl.dot() calls (calculating once instead of twice).
# 2. Fix Config alignment: block_sizes must match only the 'None' dims in hl.tile.
# 3. Use fp32 for state accumulation to maintain precision.
# 4. Streamlined gating math for B200.

# Grab all the secret NVIDIA tuning files
acf_files = glob.glob("/opt/booster_pack/chunk_fwd_h_*.acf")

# Build the config parameters properly
config_params = {
    "block_sizes": [8, 16, 32, 64], 
    "num_warps": [1, 2, 4, 8], 
    "num_stages": [1, 2, 3, 4]
}

if acf_files:
    config_params["advanced_controls_file"] = acf_files

TUNE_CONFIG = helion.Config(**config_params)

# --- HARDCODED LEADERBOARD CONFIGS ---
# block_sizes is [Tile_V] since BH is hardcoded to 1 in hl.tile
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[32], num_warps=2, num_stages=2),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[32], num_warps=4, num_stages=2),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[32], num_warps=4, num_stages=2),
    
    # Benchmark shapes
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[32], num_warps=2, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[32], num_warps=4, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[32], num_warps=4, num_stages=2),
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[25], num_warps=4, num_stages=2),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[32], num_warps=4, num_stages=2),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[32], num_warps=4, num_stages=2),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[32], num_warps=8, num_stages=2),
}

DEFAULT_CONFIG = helion.Config(block_sizes=[16], num_warps=4, num_stages=2)

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

        NT = (T + C - 1) // C
        h_out = torch.empty(B, NT, H, K, V, dtype=k.dtype, device=k.device)
        v_out = torch.empty_like(u)

        BH = B * H

        # Tile over Batch*H and V
        # BH is hardcoded to 1, V is tuned via config
        for flat, tv in hl.tile([BH, V], block_size=[1, None]):
            b_idx = flat.begin // H
            h_idx = flat.begin % H
            
            # Initialize state in FP32 for accumulation
            state = hl.zeros([K, tv], dtype=torch.float32)

            # Sequential recurrence over chunks
            for tc in hl.tile(T, block_size=C):
                chunk_idx = tc.begin // C
                t_end = (tc.begin + C) - 1 # End of chunk index

                # Store current hidden state before update
                h_out[b_idx, chunk_idx, h_idx, :, tv] = state.to(k.dtype)

                # 1. Compute projection: v_new = u - w @ h_state
                # Optimization: Single dot call
                proj = hl.dot(w[b_idx, tc, h_idx, :], state, out_dtype=torch.float32)
                
                # 2. Corrected values
                diff = (u[b_idx, tc, h_idx, tv].to(torch.float32) - proj).to(u.dtype)
                v_out[b_idx, tc, h_idx, tv] = diff

                # 3. State Decay & Update
                g_end = g[b_idx, t_end, h_idx].to(torch.float32)
                g_t = g[b_idx, tc, h_idx].to(torch.float32)
                
                # Adjust keys with gating: exp(g_last - g_t)
                alpha = torch.exp(g_end - g_t)
                k_adj = k[b_idx, tc, h_idx, :] * alpha[:, None]

                # Decay state and update: h = h * exp(g_last) + k_adj.T @ v_new
                state = state * torch.exp(g_end)
                upd = hl.dot(k_adj.T, diff.to(torch.float32), out_dtype=torch.float32)
                state = state + upd

        return h_out, v_out

    return kernel

_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}
_DEFAULT_KERNEL = _make_kernel(DEFAULT_CONFIG)

def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    kernel = _KERNELS.get((B, T, H, K, V), _DEFAULT_KERNEL)
    return kernel(k, w, u, g)
