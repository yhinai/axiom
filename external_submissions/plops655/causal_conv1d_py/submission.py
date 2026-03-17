from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, D, S, W) to optimized helion.Config objects.
# Autotune locally for each shape, then paste the best config here.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 64, 4): helion.Config(block_sizes=[1, 8], num_warps=1, num_stages=1),  # TODO: use any config that passes correctness check
    (2, 128, 128, 4): helion.Config(block_sizes=[1, 8], num_warps=1, num_stages=1),  # TODO: use any config that passes correctness check
    (1, 256, 256, 3): helion.Config(block_sizes=[1, 8], num_warps=1, num_stages=1),  # TODO: use any config that passes correctness check
    (1, 128, 64, 8): helion.Config(block_sizes=[1, 8], num_warps=1, num_stages=1),  # TODO: use any config that passes correctness check
    (4, 64, 128, 4): helion.Config(block_sizes=[1, 8], num_warps=1, num_stages=1),  # TODO: use any config that passes correctness check
    # Benchmark shapes

    (1, 768, 512, 4): helion.Config(block_sizes=[8, 128], num_warps=4, num_stages=3),
    (1, 768, 2048, 4): helion.Config(block_sizes=[8, 128], num_warps=4, num_stages=3),  
    (1, 1536, 2048, 4): helion.Config(block_sizes=[8, 64], num_warps=8, num_stages=3), 
    (1, 2560, 2048, 4): helion.Config(block_sizes=[8, 128], num_warps=2, num_stages=4),  
    (1, 2560, 4096, 4): helion.Config(block_sizes=[4, 256], num_warps=2, num_stages=2),  
    
    # (1, 768, 512, 4): helion.Config(block_sizes=[1, 8], num_warps=1, num_stages=1),  # TODO: replace with your autotuned config
    # (1, 768, 2048, 4): helion.Config(block_sizes=[1, 8], num_warps=1, num_stages=1),  # TODO: replace with your autotuned config
    # (1, 1536, 2048, 4): helion.Config(block_sizes=[1, 8], num_warps=1, num_stages=1),  # TODO: replace with your autotuned config
    # (1, 2560, 2048, 4): helion.Config(block_sizes=[1, 8], num_warps=1, num_stages=1),  # TODO: replace with your autotuned config
    # (1, 2560, 4096, 4): helion.Config(block_sizes=[1, 8], num_warps=1, num_stages=1),  # TODO: replace with your autotuned config
}


# Optional: add advanced_controls_file to your Config for extra performance (see docs).
# Autotune with autotune_search_acf to find the best ACF, then hardcode it:
#     helion.Config(..., advanced_controls_file="/opt/booster_pack/causal_conv_0.acf")


# NOTE: This is an intentionally inefficient baseline implementation.
def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        x_pad: torch.Tensor,  # (B, D, L) zero-padded input
        w: torch.Tensor,      # (D, W) filter coefficients
        b: torch.Tensor,      # (D,) additive offset
    ) -> torch.Tensor:
        B = x_pad.size(0)
        D = x_pad.size(1)
        L = x_pad.size(2)
        W = hl.specialize(w.size(1))
        N = L - W + 1

        y = torch.empty(B, D, N, dtype=x_pad.dtype, device=x_pad.device)

        for rb, rd, rs in hl.tile([B, D, N], block_size=[1, None, None]):
            bi = rb.begin
            acc = hl.zeros([rd, rs], dtype=torch.float32)
            for j in range(W):
                c = w[rd, j].to(torch.float32)
                x = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)
                acc = acc + x * c[:, None]
            acc = acc + b[rd].to(torch.float32)[:, None]
            y[rb, rd, rs] = acc[None, :, :].to(y.dtype)

        return y

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    kernel = _KERNELS[(B, D, S, W)]
    pad_zeros = torch.zeros(B, D, W - 1, dtype=x.dtype, device=x.device)
    padded = torch.cat([pad_zeros, x], dim=2)
    return kernel(padded, weight, bias)
