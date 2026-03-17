#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, D, S, W) to optimized helion.Config objects.
# Tuned for B200: larger blocks, more warps, deeper pipelining.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 64, 4): helion.Config(block_sizes=[1, 64], num_warps=4, num_stages=2),
    (2, 128, 128, 4): helion.Config(block_sizes=[1, 128], num_warps=8, num_stages=3),
    (1, 256, 256, 3): helion.Config(block_sizes=[1, 256], num_warps=8, num_stages=2),
    (1, 128, 64, 8): helion.Config(block_sizes=[1, 64], num_warps=4, num_stages=3),
    (4, 64, 128, 4): helion.Config(block_sizes=[1, 128], num_warps=8, num_stages=2),
    # Ranked shapes
    (1, 768, 512, 4): helion.Config(block_sizes=[1, 256], num_warps=16, num_stages=4),
    (1, 768, 2048, 4): helion.Config(block_sizes=[1, 256], num_warps=8, num_stages=4),
    # Benchmark shapes
    (1, 1536, 2048, 4): helion.Config(block_sizes=[1, 256], num_warps=8, num_stages=4),
    (1, 2560, 2048, 4): helion.Config(block_sizes=[1, 256], num_warps=8, num_stages=3),
    (1, 2560, 4096, 4): helion.Config(block_sizes=[1, 256], num_warps=8, num_stages=3),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        x: torch.Tensor,      # (B, D, S) original input — no host-side padding
        w: torch.Tensor,      # (D, W) filter coefficients
        b: torch.Tensor,      # (D,) additive offset
    ) -> torch.Tensor:
        B = x.size(0)
        D = x.size(1)
        S = x.size(2)
        W = hl.specialize(w.size(1))

        y = torch.empty(B, D, S, dtype=x.dtype, device=x.device)

        for rb, rd, rs in hl.tile([B, D, S], block_size=[1, None, None]):
            bi = rb.begin
            acc = hl.zeros([rd, rs], dtype=torch.float32)
            for j in range(W):
                # Causal conv: output[t] = sum_k w[k] * x[t - (W-1) + k]
                # src_idx can be negative for early positions -> implicit zero pad
                src_idx = rs.index + j - (W - 1)
                valid = src_idx >= 0
                # extra_mask with other=0 in Triton codegen gives us zero-padding
                xv = hl.load(x, [bi, rd, src_idx], extra_mask=valid).to(torch.float32)
                c = w[rd, j].to(torch.float32)
                acc = acc + xv * c[:, None]
            acc = acc + b[rd].to(torch.float32)[:, None]
            y[rb, rd, rs] = acc[None, :, :].to(y.dtype)

        return y

    return kernel


_KERNELS: dict[tuple, object] = {}


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    key = (B, D, S, W)
    if key not in _KERNELS:
        _KERNELS[key] = _make_kernel(SHAPE_CONFIGS[key])
    # No torch.cat / no HBM copy — padding handled inside the kernel
    return _KERNELS[key](x, weight, bias)
