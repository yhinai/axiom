#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius

from task import input_t, output_t
import torch
import torch.nn.functional as F
import helion
import helion.language as hl
import glob

# Optimization Strategy:
# 1. Fast zero-padding via F.pad
# 2. 3D Tiling [Batch, Depth, Sequence] for max B200 occupancy
# 3. Register/SRAM caching for weights and bias
# 4. Localized accumulation in FP32

# --- HARDCODED LEADERBOARD CONFIGS ---
# Update these with your tuned outputs after running benchmark on remote
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    (1, 64, 64, 4): helion.Config(block_sizes=[32, 64], num_warps=2, num_stages=2),
    (2, 128, 128, 4): helion.Config(block_sizes=[32, 128], num_warps=4, num_stages=2),
    (1, 256, 256, 3): helion.Config(block_sizes=[32, 128], num_warps=4, num_stages=2),
    (1, 128, 64, 8): helion.Config(block_sizes=[32, 64], num_warps=2, num_stages=2),
    (4, 64, 128, 4): helion.Config(block_sizes=[32, 64], num_warps=2, num_stages=2),
    
    # Benchmark shapes
    (1, 768, 512, 4): helion.Config(block_sizes=[32, 128], num_warps=4, num_stages=2),
    (1, 768, 2048, 4): helion.Config(block_sizes=[32, 256], num_warps=4, num_stages=3),
    (1, 1536, 2048, 4): helion.Config(block_sizes=[32, 256], num_warps=8, num_stages=3),
    (1, 2560, 2048, 4): helion.Config(block_sizes=[32, 256], num_warps=8, num_stages=3),
    (1, 2560, 4096, 4): helion.Config(block_sizes=[32, 256], num_warps=8, num_stages=3),
}

# Safe default for unknown shapes or during autotuning
DEFAULT_CONFIG = helion.Config(block_sizes=[16, 64], num_warps=4, num_stages=2)

def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        x_pad: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        B, D, L = x_pad.size(0), x_pad.size(1), x_pad.size(2)
        W = hl.specialize(w.size(1))
        N = L - W + 1
        
        # Output size is identical to original sequence length S
        y = torch.empty(B, D, N, dtype=x_pad.dtype, device=x_pad.device)

        # 3D Tiling over Batch, Channel (Depth), and Sequence
        for rb, rd, rs in hl.tile([B, D, N], block_size=[1, None, None]):
            bi = rb.begin
            
            # Pre-load bias for this tile of channels
            bias_tile = b[rd].to(torch.float32)
            
            # Local accumulator
            acc = hl.zeros([rd, rs], dtype=torch.float32)
            
            for j in range(W):
                # Load input chunk
                x_val = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)
                
                # Load weight
                w_val = w[rd, j].to(torch.float32)
                # Depthwise Multiply-Accumulate
                acc = acc + x_val * w_val[:, None]
                
            # Add bias and cast back to output type
            acc = acc + bias_tile[:, None]
            y[rb, rd, rs] = acc[None, :, :].to(y.dtype)
        return y
    return kernel

# Pre-compile kernels for hardcoded shapes
_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}
_DEFAULT_KERNEL = _make_kernel(DEFAULT_CONFIG)

def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    
    # Use F.pad for fast padding prior to kernel launch
    padded = F.pad(x, (W - 1, 0))
    
    # Dispatch to specialized or default kernel
    kernel = _KERNELS.get((B, D, S, W), _DEFAULT_KERNEL)
    return kernel(padded, weight, bias)
