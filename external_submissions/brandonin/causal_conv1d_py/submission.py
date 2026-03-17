#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, D, S, W) to optimized helion.Config objects.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes — simple configs that pass correctness
    (1, 64, 64, 4): helion.Config(block_sizes=[1, 64, 64], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/causal_conv_0.acf"),
    (2, 128, 128, 4): helion.Config(block_sizes=[1, 128, 128], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/causal_conv_0.acf"),
    (1, 256, 256, 3): helion.Config(block_sizes=[1, 256, 256], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/causal_conv_0.acf"),
    (1, 128, 64, 8): helion.Config(block_sizes=[1, 128, 64], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/causal_conv_0.acf"),
    (4, 64, 128, 4): helion.Config(block_sizes=[1, 64, 128], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/causal_conv_0.acf"),
    # Benchmark shapes — autotuned configs (ACF sweep)
    (1, 1536, 2048, 4): helion.Config(
        advanced_controls_file='/opt/booster_pack/causal_conv_0.acf',
        block_sizes=[1, 2, 256],
        indexing=['pointer', 'pointer', 'pointer', 'pointer'],
        l2_groupings=[1],
        load_eviction_policies=['', 'last', ''],
        loop_orders=[[0, 1, 2]],
        num_stages=1, num_warps=1, pid_type='flat',
        range_flattens=[None, None], range_multi_buffers=[None, True],
        range_num_stages=[0, 0], range_unroll_factors=[0, 0],
        range_warp_specializes=[None, False], static_ranges=[False],
    ),
    (1, 2560, 2048, 4): helion.Config(
        advanced_controls_file='/opt/booster_pack/causal_conv_0.acf',
        block_sizes=[1, 1, 512],
        indexing=['pointer', 'pointer', 'pointer', 'pointer'],
        l2_groupings=[1],
        load_eviction_policies=['', '', ''],
        loop_orders=[[0, 2, 1]],
        num_stages=1, num_warps=1, pid_type='flat',
        range_flattens=[None, None], range_multi_buffers=[None, None],
        range_num_stages=[0, 0], range_unroll_factors=[0, 0],
        range_warp_specializes=[None, None], static_ranges=[False],
    ),
    (1, 2560, 4096, 4): helion.Config(
        block_sizes=[1, 1, 1024],
        indexing=['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'],
        l2_groupings=[2],
        load_eviction_policies=['', '', 'last'],
        loop_orders=[[2, 1, 0]],
        num_stages=2, num_warps=1, pid_type='flat',
        range_flattens=[None, True], range_multi_buffers=[None, None],
        range_num_stages=[0, 3], range_unroll_factors=[0, 4],
        range_warp_specializes=[None, None], static_ranges=[False],
    ),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        x: torch.Tensor,     # (B, D, S) UNPADDED input
        w: torch.Tensor,     # (D, W) filter coefficients
        b: torch.Tensor,     # (D,) additive offset
    ) -> torch.Tensor:
        B = x.size(0)
        D = x.size(1)
        S = x.size(2)
        W = hl.specialize(w.size(1))

        y = torch.empty(B, D, S, dtype=x.dtype, device=x.device)

        for rb, rd, rs in hl.tile([B, D, S]):
            bi = rb.begin
            acc = hl.zeros([rd, rs], dtype=torch.float32)
            for j in range(W):
                c = w[rd, j].to(torch.float32)
                # Causal indexing: need x at position (t - W + 1 + j)
                # For output t, read x[t - (W-1) + j]. Negative => zero.
                src_idx = rs.index + j - (W - 1)
                safe_idx = torch.clamp(src_idx, min=0)
                x_val = hl.load(x, [bi, rd, safe_idx]).to(torch.float32)
                # Zero out where original index was negative (causal padding)
                mask = src_idx >= 0
                x_val = torch.where(mask, x_val, 0.0)
                acc = acc + x_val * c[:, None]
            acc = acc + b[rd].to(torch.float32)[:, None]
            y[rb, rd, rs] = acc[None, :, :].to(y.dtype)

        return y

    return kernel


_KERNELS: dict = {}


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    shape = (B, D, S, W)
    if shape not in _KERNELS:
        _KERNELS[shape] = _make_kernel(SHAPE_CONFIGS[shape])
    # NO torch.cat/zeros padding - fused into kernel via causal indexing!
    return _KERNELS[shape](x, weight, bias)
