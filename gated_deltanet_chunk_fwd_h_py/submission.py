#!POPCORN leaderboard gated_deltanet_chunk_fwd_h
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
# V block size controlled via block_sizes[1], gate applied to diff (not k) for fewer FLOPs
_DEFAULT = helion.Config(block_sizes=[1, 8], indexing=['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], l2_groupings=[1], load_eviction_policies=['first', '', '', '', ''], loop_orders=[[1, 0]], num_stages=3, num_warps=4, pid_type='flat', range_flattens=[None, True], range_multi_buffers=[None, False], range_num_stages=[0, 3], range_unroll_factors=[0, 0], range_warp_specializes=[None, None])

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): _DEFAULT,
    (2, 128, 4, 64, 64): _DEFAULT,
    (1, 256, 4, 64, 128): _DEFAULT,
    # Benchmark shapes — per-shape tuning targets
    (1, 64, 1, 64, 64): _DEFAULT,    # BH=1, tiny: needs max parallelism
    (2, 512, 3, 64, 64): _DEFAULT,   # BH=6, medium: balance parallelism/efficiency
    (2, 1024, 3, 64, 64): _DEFAULT,  # BH=6, large: same as medium
}

# exp -> exp2 constant: exp(x) = exp2(x * LOG2E)
_LOG2E = 1.4426950408889634


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="tf32", config=config)
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

        for flat, tv in hl.tile([BH, V]):
            b_idx = flat.begin // H
            h_idx = flat.begin % H
            state = hl.zeros([K, tv], dtype=torch.float32)

            for tc in hl.tile(T, block_size=C):
                chunk_idx = tc.begin // C
                t_end = min(tc.begin + C, T) - 1

                # Store current state as h_out for this chunk
                h_out[b_idx, chunk_idx, h_idx, :, tv] = state.to(k.dtype)

                # Compute v_new = u - w @ state
                proj = hl.dot(
                    w[b_idx, tc, h_idx, :], state, out_dtype=torch.float32
                )
                diff = u[b_idx, tc, h_idx, tv].to(torch.float32) - proj
                v_out[b_idx, tc, h_idx, tv] = diff.to(u.dtype)

                # Update state: state = state * exp(g_end) + k^T @ (v_new * gate)
                g_end = g[b_idx, t_end, h_idx].to(torch.float32)
                g_t = g[b_idx, tc, h_idx].to(torch.float32)
                valid = tc.index < T
                # Use exp2 instead of exp: exp(x) = exp2(x * log2(e))
                alpha = torch.where(valid, torch.exp2((g_end - g_t) * _LOG2E), 0.0)
                diff_gated = diff * alpha[:, None]  # gate diff not k: [C,tv] not [C,K]

                # Fused state decay + accumulation
                state = state * torch.exp2(g_end * _LOG2E)
                state = hl.dot(
                    k[b_idx, tc, h_idx, :].T, diff_gated, acc=state, out_dtype=torch.float32
                )

        return h_out, v_out

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(k, w, u, g)
