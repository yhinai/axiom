#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius
from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, D, S, W) to optimized helion.Config objects.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Benchmark shapes only — test shapes use fallback (default config)
    (1, 1536, 2048, 4): helion.Config(block_sizes=[32, 32], indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', '', '', '', '', '', '', '', ''], loop_orders=[[0, 1]], num_stages=1, num_warps=4, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 0], range_unroll_factors=[0, 0], range_warp_specializes=[None, None], static_ranges=[False]),
    (1, 2560, 2048, 4): helion.Config(block_sizes=[32, 32], indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', '', '', '', '', '', '', '', ''], loop_orders=[[0, 1]], num_stages=1, num_warps=16, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 1], range_unroll_factors=[0, 0], range_warp_specializes=[None, None], static_ranges=[False]),
    (1, 2560, 4096, 4): helion.Config(advanced_controls_file='/opt/booster_pack/causal_conv_1.acf', block_sizes=[1, 2048], indexing=['pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer'], l2_groupings=[1], load_eviction_policies=['last', 'last', 'last', 'first', 'first', 'first', '', '', 'last'], loop_orders=[[1, 0]], num_stages=3, num_warps=8, pid_type='flat', range_flattens=[None, None], range_multi_buffers=[None, None], range_num_stages=[0, 4], range_unroll_factors=[0, 3], range_warp_specializes=[None, None], static_ranges=[False]),
}


@helion.kernel(autotune_effort="none")
def causal_conv1d_kernel(
    x: torch.Tensor,       # [B, D, S]
    weight: torch.Tensor,  # [D, W]
    bias: torch.Tensor,    # [D]
) -> torch.Tensor:
    B, D, S = x.size()
    W = hl.specialize(weight.size(1))
    out = torch.empty_like(x)

    for tile_d, tile_s in hl.tile([D, S]):
        w = [weight[tile_d, k][:, None] for k in hl.static_range(W)]

        for b in range(B):
            acc = hl.zeros([tile_d, tile_s], dtype=torch.float32)
            for k in hl.static_range(W):
                idx = tile_s.index + k - (W - 1)
                x_vals = hl.load(x, [b, tile_d, idx], extra_mask=idx >= 0)
                acc = acc + w[k] * x_vals
            out[b, tile_d, tile_s] = acc + bias[tile_d][:, None]

    return out


# Pre-compile and warm up a runner for each shape
_RUNNERS: dict[tuple, object] = {}
for (_B, _D, _S, _W), _cfg in SHAPE_CONFIGS.items():
    _x = torch.empty(_B, _D, _S, dtype=torch.float32, device="cuda")
    _weight = torch.empty(_D, _W, dtype=torch.float32, device="cuda")
    _bias = torch.empty(_D, dtype=torch.float32, device="cuda")
    _bound = causal_conv1d_kernel.bind((_x, _weight, _bias))
    _runner = _bound.compile_config(_cfg)
    _runner(_x, _weight, _bias)
    _RUNNERS[(_B, _D, _S, _W)] = _runner
torch.cuda.synchronize()


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]

    key = (B, D, S, W)
    if key in _RUNNERS:
        return _RUNNERS[key](x, weight, bias)
    return causal_conv1d_kernel(x, weight, bias)
