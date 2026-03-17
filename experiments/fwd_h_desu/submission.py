#!POPCORN leaderboard gated_deltanet_chunk_fwd_h
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# VG6: keep WL1's V-tiling/persistent structure, but run both recurrent dots
# with fp32 inputs so Helion can use TF32-style tensor cores without the long-
# sequence drift caused by bf16 update inputs.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[32], num_warps=4, num_stages=2),
    # Benchmark shapes
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[64], num_warps=8, num_stages=2),
    # Extra shapes kept from earlier tuning tables
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[32], num_warps=8, num_stages=2),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[32], num_warps=8, num_stages=2),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[32], num_warps=8, num_stages=2),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[64], num_warps=8, num_stages=3),
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        k_fp32: torch.Tensor,  # [B, T, H, K]
        w_fp32: torch.Tensor,  # [B, T, H, K]
        u_fp32: torch.Tensor,  # [B, T, H, V]
        g_fp32: torch.Tensor,  # [B, T, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k_fp32.shape
        V = u_fp32.shape[-1]
        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)

        NT = T // C
        h_out = torch.empty(B, NT, H, K, V, dtype=torch.float32, device=u_fp32.device)
        v_out = torch.empty(B, T, H, V, dtype=torch.float32, device=u_fp32.device)

        BH = B * H

        for flat, tv in hl.tile([BH, V], block_size=[1, None]):
            b_idx = flat.begin // H
            h_idx = flat.begin % H

            state = hl.zeros([K, tv], dtype=torch.float32)

            for chunk_idx in range(NT):
                t0 = chunk_idx * C
                t1 = t0 + C
                t_end = t1 - 1

                h_out[b_idx, chunk_idx, h_idx, :, tv] = state

                w_tile = w_fp32[b_idx, t0:t1, h_idx, :]
                u_tile = u_fp32[b_idx, t0:t1, h_idx, tv]
                g_tile = g_fp32[b_idx, t0:t1, h_idx]
                g_end = g_fp32[b_idx, t_end, h_idx].to(torch.float32)

                # Precision-critical recurrent projection: keep both inputs fp32.
                proj = hl.dot(w_tile, state, acc=hl.zeros([C, tv], dtype=torch.float32))
                diff = u_tile.to(torch.float32) - proj
                v_out[b_idx, t0:t1, h_idx, tv] = diff

                decay = torch.exp(g_end)
                alpha = torch.exp(g_end - g_tile.to(torch.float32))
                v_gated = diff * alpha[:, None]

                # Recurrent update stays fp32 to avoid error accumulation over chunks.
                k_tile = k_fp32[b_idx, t0:t1, h_idx, :]
                state = hl.dot(k_tile.T, v_gated, acc=state * decay)

        return h_out, v_out

    return kernel


_KERNELS: dict[tuple, object] = {}


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    key = (B, T, H, K, V)

    if key not in _KERNELS:
        _KERNELS[key] = _make_kernel(SHAPE_CONFIGS[key])

    return _KERNELS[key](k, w, u, g)
