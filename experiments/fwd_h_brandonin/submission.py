#!POPCORN leaderboard gated_deltanet_chunk_fwd_h
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
# block_sizes has 1 entry for the V tile dimension (the None in block_size=[1, None])
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/chunk_fwd_h_0.acf"),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/chunk_fwd_h_0.acf"),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[64], num_warps=4, num_stages=2, advanced_controls_file="/opt/booster_pack/chunk_fwd_h_0.acf"),
    # Benchmark shapes — autotuned configs (quick effort, ACF sweep)
    (1, 64, 1, 64, 64): helion.Config(
        advanced_controls_file='/opt/booster_pack/chunk_fwd_h_0.acf',
        block_sizes=[4],
        indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'],
        l2_groupings=[1], load_eviction_policies=['', '', '', '', ''],
        loop_orders=[[0, 1]], num_stages=1, num_warps=16, pid_type='flat',
        range_flattens=[None, None], range_multi_buffers=[None, None],
        range_num_stages=[0, 0], range_unroll_factors=[0, 1],
        range_warp_specializes=[None, None], static_ranges=[False],
    ),
    (2, 512, 3, 64, 64): helion.Config(
        advanced_controls_file='/opt/booster_pack/chunk_fwd_h_0.acf',
        block_sizes=[8],
        indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'],
        l2_groupings=[1], load_eviction_policies=['last', '', '', '', ''],
        loop_orders=[[0, 1]], num_stages=3, num_warps=4, pid_type='flat',
        range_flattens=[None, None], range_multi_buffers=[None, False],
        range_num_stages=[0, 0], range_unroll_factors=[0, 0],
        range_warp_specializes=[None, None], static_ranges=[False],
    ),
    (2, 1024, 3, 64, 64): helion.Config(
        advanced_controls_file='/opt/booster_pack/chunk_fwd_h_1.acf',
        block_sizes=[8],
        indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'],
        l2_groupings=[1], load_eviction_policies=['', '', '', '', ''],
        loop_orders=[[0, 1]], num_stages=5, num_warps=4, pid_type='flat',
    ),
    # Extra shapes (fallback)
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[8], num_warps=4, num_stages=3, advanced_controls_file="/opt/booster_pack/chunk_fwd_h_0.acf"),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[8], num_warps=4, num_stages=3, advanced_controls_file="/opt/booster_pack/chunk_fwd_h_0.acf"),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[8], num_warps=4, num_stages=3, advanced_controls_file="/opt/booster_pack/chunk_fwd_h_0.acf"),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[8], num_warps=4, num_stages=3, advanced_controls_file="/opt/booster_pack/chunk_fwd_h_0.acf"),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="tf32", fast_math=True, config=config)
    def kernel(
        k: torch.Tensor,   # [B, T, H, K]
        w: torch.Tensor,   # [B, T, H, K]
        u: torch.Tensor,   # [B, T, H, V]
        g: torch.Tensor,   # [B, T, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = u.shape[-1]
        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)

        NT = (T + C - 1) // C
        h_out = torch.empty(B, NT, H, K, V, dtype=k.dtype, device=k.device)
        v_out = torch.empty_like(u)

        BH = B * H

        for flat, tv in hl.tile([BH, V], block_size=[1, None]):
            b_idx = flat.begin // H
            h_idx = flat.begin % H
            state = hl.zeros([K, tv], dtype=torch.float32)

            for tc in hl.tile(T, block_size=C):
                chunk_idx = tc.begin // C
                t_end = min(tc.begin + C, T) - 1

                # Store current state
                h_out[b_idx, chunk_idx, h_idx, :, tv] = state.to(k.dtype)

                # Compute v_new = u - w @ h_state
                proj = hl.dot(
                    w[b_idx, tc, h_idx, :], state, out_dtype=torch.float32
                )
                diff = u[b_idx, tc, h_idx, tv].to(torch.float32) - proj
                v_out[b_idx, tc, h_idx, tv] = diff.to(u.dtype)

                # Gate: v_gated[t] = v_new[t] * exp(g_last - g[t])
                g_end = g[b_idx, t_end, h_idx].to(torch.float32)
                g_t = g[b_idx, tc, h_idx].to(torch.float32)
                valid = tc.index < T
                alpha = torch.where(valid, torch.exp(g_end - g_t), 0.0)
                k_adj = k[b_idx, tc, h_idx, :] * alpha[:, None]

                # Decay state and update
                state = state * torch.exp(g_end)
                state = hl.dot(k_adj.T, diff, acc=state)

        return h_out, v_out

    return kernel


_KERNELS: dict = {}


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    shape = (B, T, H, K, V)
    if shape not in _KERNELS:
        _KERNELS[shape] = _make_kernel(SHAPE_CONFIGS[shape])
    return _KERNELS[shape](k, w, u, g)
