#!POPCORN leaderboard gated_deltanet_recompute_w_u
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Single ACF-autotuned config for all shapes (avoids per-shape compilation overhead)
_TUNED = helion.Config(advanced_controls_file='/opt/booster_pack/recompute_w_u_fwd_0.acf', block_sizes=[], indexing=['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'pointer', 'pointer', 'pointer'], l2_groupings=[1], load_eviction_policies=['', '', '', '', ''], loop_orders=[[0, 1]], num_sm_multiplier=1, num_stages=1, num_warps=8, pid_type='persistent_interleaved', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None])

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): _TUNED,
    (2, 128, 4, 64, 64): _TUNED,
    (1, 256, 4, 64, 128): _TUNED,
    (1, 64, 1, 128, 128): _TUNED,
    (2, 128, 2, 100, 100): _TUNED,
    # Benchmark shapes
    (1, 64, 1, 64, 64): _TUNED,
    (2, 512, 3, 64, 64): _TUNED,
    (2, 1024, 3, 64, 64): _TUNED,
    (3, 1024, 4, 100, 100): _TUNED,
    (4, 1024, 4, 128, 128): _TUNED,
    (2, 1536, 4, 128, 128): _TUNED,
    (4, 2048, 8, 64, 64): _TUNED,
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        g: torch.Tensor,
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

            beta_vals = beta[b_idx, rt, h_idx].to(torch.float32)
            g_vals = g[b_idx, rt, h_idx].to(torch.float32)
            k_chunk = k[b_idx, rt, h_idx, :].to(torch.float32)
            v_chunk = v[b_idx, rt, h_idx, :].to(torch.float32)
            A_chunk = A[b_idx, rt, h_idx, :].to(torch.float32)

            # exp2 is a single GPU instruction vs multi-instruction exp
            LOG2E = 1.4426950408889634
            k_scaled = k_chunk * (beta_vals * torch.exp2(g_vals * LOG2E))[:, None]
            v_scaled = v_chunk * beta_vals[:, None]

            w_result = hl.dot(A_chunk, k_scaled, out_dtype=torch.float32)
            u_result = hl.dot(A_chunk, v_scaled, out_dtype=torch.float32)

            w_out[b_idx, rt, h_idx, :] = w_result.to(k.dtype)
            u_out[b_idx, rt, h_idx, :] = u_result.to(v.dtype)

        return w_out, u_out

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    shape = (B, T, H, K, V)
    if shape not in _KERNELS:
        _KERNELS[shape] = _make_kernel(_TUNED)
    kernel = _KERNELS[shape]
    return kernel(k, v, beta, A, g)
