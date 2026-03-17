#!POPCORN leaderboard gated_deltanet_chunk_fwd_h
#!POPCORN gpu B200_Nebius

from task import input_t, output_t
import torch
import helion
import helion.language as hl

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[16], indexing=['block_ptr', 'pointer', 'pointer', 'pointer', 'block_ptr', 'pointer', 'block_ptr'], l2_groupings=[1], load_eviction_policies=['', 'first', 'last', 'first', 'first'], loop_orders=[[1, 0]], num_stages=1, num_warps=8, pid_type='flat'),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[16], indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'block_ptr', 'pointer'], l2_groupings=[4], load_eviction_policies=['first', 'first', '', 'first', ''], loop_orders=[[1, 0]], num_stages=5, num_warps=4, pid_type='flat'),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[64], indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'block_ptr', 'pointer'], l2_groupings=[4], load_eviction_policies=['first', 'first', '', 'first', ''], loop_orders=[[1, 0]], num_stages=5, num_warps=4, pid_type='flat'),
    # Benchmark shapes
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[64], indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'block_ptr', 'pointer'], l2_groupings=[4], load_eviction_policies=['first', 'first', '', 'first', ''], loop_orders=[[1, 0]], num_stages=5, num_warps=4, pid_type='flat'),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[32], loop_orders=[[0, 1]], l2_groupings=[32], load_eviction_policies=['last', '', '', 'first', 'last'], num_warps=8, num_stages=4, indexing=['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], pid_type='flat'),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[16], loop_orders=[[1, 0]], l2_groupings=[1], load_eviction_policies=['', '', 'last', 'last', ''], num_warps=4, num_stages=3, indexing=['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer'], pid_type='flat'),
    # Ranked shapes
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[20], loop_orders=[[1, 0]], l2_groupings=[16], load_eviction_policies=['', '', '', '', ''], num_warps=8, num_stages=5, indexing=['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], pid_type='flat'),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[32], loop_orders=[[1, 0]], l2_groupings=[8], load_eviction_policies=['last', '', 'first', 'last', ''], num_warps=4, num_stages=4, indexing=['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], pid_type='flat'),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[32], loop_orders=[[1, 0]], l2_groupings=[4], load_eviction_policies=['first', 'first', '', 'first', ''], num_warps=4, num_stages=5, indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'block_ptr', 'pointer'], pid_type='flat'),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[32], loop_orders=[[1, 0]], l2_groupings=[4], load_eviction_policies=['first', 'first', '', 'first', ''], num_warps=8, num_stages=5, indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'block_ptr', 'pointer'], pid_type='flat'),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
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

        NT = T // C
        h_out = torch.empty(B, NT, H, K, V, dtype=k.dtype, device=k.device)
        v_out = torch.empty_like(u)

        block_v = hl.register_block_size(V)
        BH = B * H

        for flat, tv in hl.tile([BH, V], block_size=[1, block_v]):
            b_idx = flat.begin // H
            h_idx = flat.begin % H
            state = hl.zeros([K, tv], dtype=torch.float32)

            for tc in hl.tile(T, block_size=C):
                chunk_idx = tc.begin // C
                t_end = tc.begin + C - 1

                h_out[b_idx, chunk_idx, h_idx, :, tv] = state.to(k.dtype)

                proj = hl.dot(w[b_idx, tc, h_idx, :], state, out_dtype=torch.float32)
                diff = u[b_idx, tc, h_idx, tv].to(torch.float32) - proj
                v_out[b_idx, tc, h_idx, tv] = diff.to(u.dtype)

                g_end = g[b_idx, t_end, h_idx]
                alpha = torch.exp(g_end - g[b_idx, tc, h_idx])
                diff_gated = diff * alpha[:, None]

                state = state * torch.exp(g_end)
                state = hl.dot(k[b_idx, tc, h_idx, :].T, diff_gated, acc=state)

        return h_out, v_out

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(k, w, u, g)
