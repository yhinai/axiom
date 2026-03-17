#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Autotuned config from previous run — used as starting point for all shapes
# Will be replaced with per-shape autotuned configs after B200 autotuning
_TUNED = helion.Config(block_sizes=[1, 1024], indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], l2_groupings=[8], load_eviction_policies=['first', '', 'first'], loop_orders=[[0, 2, 1]], num_stages=7, num_warps=2, pid_type='flat', range_flattens=[None, False], range_multi_buffers=[None, True], range_num_stages=[0, 0], range_unroll_factors=[0, 0], range_warp_specializes=[None, False], static_ranges=[False])
_DEFAULT = helion.Config(block_sizes=[1, 256], num_warps=4, num_stages=1)

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes — use _DEFAULT for fast compilation
    (1, 64, 64, 4): _DEFAULT,
    (2, 128, 128, 4): _DEFAULT,
    (1, 256, 256, 3): _DEFAULT,
    (1, 128, 64, 8): _DEFAULT,
    (4, 64, 128, 4): _DEFAULT,
    # Benchmark shapes — use _TUNED (was using _DEFAULT before, big perf miss)
    (1, 768, 512, 4): _TUNED,
    (1, 768, 2048, 4): _TUNED,
    (1, 1536, 2048, 4): _TUNED,
    (1, 2560, 2048, 4): _TUNED,
    (1, 2560, 4096, 4): _TUNED,
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        # x shape: [B, D, S] — original unpadded input
        B = x.size(0)
        D = x.size(1)
        S = x.size(2)
        # W = filter width, compile-time constant for unrolling
        W = hl.specialize(w.size(1))

        # Output has same shape as input: [B, D, S]
        y = torch.empty(B, D, S, dtype=x.dtype, device=x.device)

        # Tile over batch (fixed 1), channels, and sequence length
        for rb, rd, rs in hl.tile([B, D, S], block_size=[1, None, None]):
            bi = rb.begin
            # Accumulator for the convolution sum, one per (channel, time) pair
            acc = hl.zeros([rd, rs], dtype=torch.float32)
            # Unrolled loop over filter taps (W is typically 4)
            for j in range(W):
                # Load the j-th filter weight for each channel
                coeff = w[rd, j].to(torch.float32)
                # Compute index into original x: for output position s,
                # we need x[b, d, s + j - (W-1)] (causal: only past values)
                idx = rs.index + j - (W - 1)
                # Clamp to 0 so pointer arithmetic stays in-bounds,
                # then mask out the invalid (negative) positions to get 0.0
                safe_idx = torch.where(idx >= 0, idx, torch.zeros_like(idx))
                x_val = hl.load(x, [bi, rd, safe_idx], extra_mask=idx >= 0).to(torch.float32)
                # Multiply-accumulate: weight * input value
                acc = acc + x_val * coeff[:, None]
            # Add per-channel bias
            acc = acc + b[rd].to(torch.float32)[:, None]
            # Write output tile
            y[rb, rd, rs] = acc[None, :, :].to(y.dtype)

        return y

    return kernel


# Pre-compile kernels for all known shapes at import time
_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    shape = (B, D, S, W)
    # Look up pre-compiled kernel for this shape, or compile with _TUNED as fallback
    if shape not in _KERNELS:
        _KERNELS[shape] = _make_kernel(_TUNED)
    kernel = _KERNELS[shape]
    # No host-side padding — boundary handled inside kernel via masked loads
    return kernel(x, weight, bias)
