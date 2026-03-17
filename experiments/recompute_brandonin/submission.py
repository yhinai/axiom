#!POPCORN leaderboard gated_deltanet_recompute_w_u
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
# block_sizes=[] because all tile block sizes are specified explicitly
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/recompute_w_u_fwd_2.acf"),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/recompute_w_u_fwd_2.acf"),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/recompute_w_u_fwd_2.acf"),
    # Benchmark shapes — autotuned configs (quick effort, ACF sweep)
    (1, 64, 1, 64, 64): helion.Config(
        advanced_controls_file='/opt/booster_pack/recompute_w_u_fwd_0.acf',
        block_sizes=[],
        indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'],
        l2_groupings=[1], load_eviction_policies=['', '', '', '', ''],
        loop_orders=[[0, 1]], num_sm_multiplier=2, num_stages=1, num_warps=8,
        pid_type='persistent_blocked',
        range_flattens=[None], range_multi_buffers=[None],
        range_num_stages=[0], range_unroll_factors=[0],
    ),
    (2, 512, 3, 64, 64): helion.Config(
        advanced_controls_file='/opt/booster_pack/recompute_w_u_fwd_2.acf',
        block_sizes=[],
        indexing=['pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer'],
        l2_groupings=[2], load_eviction_policies=['', '', '', '', ''],
        loop_orders=[[0, 1]], num_stages=1, num_warps=4, pid_type='flat',
        range_flattens=[None], range_multi_buffers=[None],
        range_num_stages=[0], range_unroll_factors=[0],
    ),
    (2, 1024, 3, 64, 64): helion.Config(
        advanced_controls_file='/opt/booster_pack/recompute_w_u_fwd_0.acf',
        block_sizes=[],
        indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'],
        l2_groupings=[1], load_eviction_policies=['', '', '', 'first', ''],
        loop_orders=[[0, 1]], num_sm_multiplier=2, num_stages=1, num_warps=8,
        pid_type='persistent_blocked',
        range_flattens=[None], range_multi_buffers=[None],
        range_num_stages=[0], range_unroll_factors=[2],
    ),
    # Extra shapes (fallback)
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/recompute_w_u_fwd_2.acf"),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/recompute_w_u_fwd_2.acf"),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/recompute_w_u_fwd_2.acf"),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/recompute_w_u_fwd_2.acf"),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="tf32", fast_math=True, config=config)
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

            # Load A block: [C, C]
            a_block = A[b_idx, rt, h_idx, :].to(torch.float32)

            # Load and scale inputs
            beta_vals = beta[b_idx, rt, h_idx].to(torch.float32)
            g_vals = g[b_idx, rt, h_idx].to(torch.float32)

            # u = A @ (v * beta[:, None])
            v_vals = v[b_idx, rt, h_idx, :].to(torch.float32)
            v_scaled = v_vals * beta_vals[:, None]
            u_result = hl.dot(a_block, v_scaled, out_dtype=torch.float32)

            # w = A @ (k * (beta * exp(g))[:, None])
            k_vals = k[b_idx, rt, h_idx, :].to(torch.float32)
            k_scaled = k_vals * (beta_vals * torch.exp(g_vals))[:, None]
            w_result = hl.dot(a_block, k_scaled, out_dtype=torch.float32)

            w_out[b_idx, rt, h_idx, :] = w_result.to(k.dtype)
            u_out[b_idx, rt, h_idx, :] = u_result.to(v.dtype)

        return w_out, u_out

    return kernel


_KERNELS: dict = {}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    shape = (B, T, H, K, V)
    if shape not in _KERNELS:
        _KERNELS[shape] = _make_kernel(SHAPE_CONFIGS[shape])
    return _KERNELS[shape](k, v, beta, A, g)
