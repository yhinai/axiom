#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, D, S, W) to optimized helion.Config objects.
# Autotuned config from B200 (full effort, 710 configs, min=0.0205ms, with ACF booster)
_TUNED = helion.Config(advanced_controls_file='/opt/booster_pack/causal_conv_1.acf', block_sizes=[1, 1024], indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], l2_groupings=[8], load_eviction_policies=['first', '', 'first'], loop_orders=[[0, 2, 1]], num_stages=7, num_warps=2, pid_type='flat', range_flattens=[None, False], range_multi_buffers=[None, True], range_num_stages=[0, 0], range_unroll_factors=[0, 0], range_warp_specializes=[None, False], static_ranges=[False])
_DEFAULT = helion.Config(block_sizes=[1, 256], num_warps=4, num_stages=1)

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 64, 4): _DEFAULT,
    (2, 128, 128, 4): _DEFAULT,
    (1, 256, 256, 3): _DEFAULT,
    (1, 128, 64, 8): _DEFAULT,
    (4, 64, 128, 4): _DEFAULT,
    # Benchmark shapes
    (1, 768, 512, 4): _TUNED,
    (1, 768, 2048, 4): _TUNED,
    (1, 1536, 2048, 4): _TUNED,
    (1, 2560, 2048, 4): _TUNED,
    (1, 2560, 4096, 4): _TUNED,
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        x_pad: torch.Tensor,  # (B, D, L) zero-padded input
        w: torch.Tensor,      # (D, W) filter coefficients
        b: torch.Tensor,      # (D,) additive bias
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
                coeff = w[rd, j].to(torch.float32)
                x_val = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)
                acc = acc + x_val * coeff[:, None]
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
