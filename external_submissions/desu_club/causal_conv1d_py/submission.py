#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius

import os

os.environ["ENABLE_TILE"] = "1"
os.environ["HELION_BACKEND"] = "tileir"

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# VG10: TileIR with per-shape best configs from LFBO autotuning on B200.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Best configs from causal_conv1d_py_benchmark_VG2.log
    (1, 1536, 2048, 4): helion.Config(
        block_sizes=[1, 1024], indexing=['pointer', 'pointer', 'pointer', 'pointer'],
        l2_groupings=[16], load_eviction_policies=['', '', ''],
        loop_orders=[[1, 2, 0]], num_ctas=1, num_stages=5, num_warps=4,
        occupancy=2, pid_type='flat', range_flattens=[], range_multi_buffers=[],
        range_num_stages=[], range_unroll_factors=[], range_warp_specializes=[],
    ),
    (1, 2560, 2048, 4): helion.Config(
        block_sizes=[1, 2048], indexing=['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer'],
        l2_groupings=[8], load_eviction_policies=['', '', ''],
        loop_orders=[[0, 2, 1]], num_ctas=1, num_stages=8, num_warps=4,
        occupancy=4, pid_type='flat', range_flattens=[], range_multi_buffers=[],
        range_num_stages=[], range_unroll_factors=[], range_warp_specializes=[],
    ),
    (1, 2560, 4096, 4): helion.Config(
        block_sizes=[2, 4096], indexing=['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'],
        l2_groupings=[1], load_eviction_policies=['', '', ''],
        loop_orders=[[0, 2, 1]], num_ctas=2, num_stages=7, num_warps=4,
        occupancy=2, pid_type='flat', range_flattens=[], range_multi_buffers=[],
        range_num_stages=[], range_unroll_factors=[], range_warp_specializes=[],
    ),
    # Test shapes — use a safe default
    (1, 64, 64, 4): helion.Config(block_sizes=[1, 64], num_ctas=1, occupancy=4, num_stages=4, indexing="tensor_descriptor"),
    (2, 128, 128, 4): helion.Config(block_sizes=[1, 128], num_ctas=1, occupancy=4, num_stages=4, indexing="tensor_descriptor"),
    (1, 256, 256, 3): helion.Config(block_sizes=[1, 256], num_ctas=1, occupancy=4, num_stages=4, indexing="tensor_descriptor"),
    (1, 128, 64, 8): helion.Config(block_sizes=[1, 64], num_ctas=1, occupancy=4, num_stages=4, indexing="tensor_descriptor"),
    (4, 64, 128, 4): helion.Config(block_sizes=[1, 128], num_ctas=1, occupancy=4, num_stages=4, indexing="tensor_descriptor"),
    # Other benchmark shapes — use VG2 default
    (1, 768, 512, 4): helion.Config(block_sizes=[1, 512], num_ctas=1, occupancy=4, num_stages=5, indexing="tensor_descriptor"),
    (1, 768, 2048, 4): helion.Config(block_sizes=[1, 1024], num_ctas=1, occupancy=4, num_stages=5, indexing="tensor_descriptor"),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        x_pad: torch.Tensor,  # [B, D, S+W-1] fp32
        w: torch.Tensor,       # [D, W] fp32
        b: torch.Tensor,       # [D] fp32
    ) -> torch.Tensor:
        B = x_pad.size(0)
        D = x_pad.size(1)
        L = x_pad.size(2)
        W = hl.specialize(w.size(1))
        S = L - W + 1

        y = torch.empty(B, D, S, dtype=x_pad.dtype, device=x_pad.device)

        for rb, rd, rs in hl.tile([B, D, S], block_size=[1, None, None]):
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


_KERNELS: dict[tuple, object] = {}


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    key = (B, D, S, W)

    if key not in _KERNELS:
        _KERNELS[key] = _make_kernel(SHAPE_CONFIGS[key])

    pad_zeros = torch.zeros(B, D, W - 1, dtype=x.dtype, device=x.device)
    padded = torch.cat([pad_zeros, x], dim=2)

    return _KERNELS[key](padded, weight, bias)
