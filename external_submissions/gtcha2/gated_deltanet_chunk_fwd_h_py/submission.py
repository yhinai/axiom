from task import input_t, output_t

import torch
import helion
import helion.language as hl


SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[16], num_warps=8, num_stages=1),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[8], num_warps=4, num_stages=3),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[8], num_warps=4, num_stages=3),
    # Benchmark shapes (autotuned winners)
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[16], num_warps=8, num_stages=1),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[8], num_warps=4, num_stages=3),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[8], num_warps=4, num_stages=3),
    # V=100/128 shapes — using best V=64 pattern, autotune later for these
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[8], num_warps=4, num_stages=3),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[8], num_warps=4, num_stages=3),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[8], num_warps=4, num_stages=3),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[8], num_warps=4, num_stages=3),
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

                # 1. Store current state
                h_out[b_idx, chunk_idx, h_idx, :, tv] = state.to(k.dtype)

                # 2. v_new = u - w @ h  (w @ state is [C,K] @ [K,V] → [C,V])
                proj = hl.dot(
                    w[b_idx, tc, h_idx, :], state, out_dtype=torch.float32
                )
                diff = u[b_idx, tc, h_idx, tv].to(torch.float32) - proj
                v_out[b_idx, tc, h_idx, tv] = diff.to(u.dtype)

                # 3. Gate: v_gated = v_new * exp(g_last - g[t])
                g_end = g[b_idx, t_end, h_idx].to(torch.float32)
                g_t = g[b_idx, tc, h_idx].to(torch.float32)
                valid = tc.index < T
                alpha = torch.where(valid, torch.exp(g_end - g_t), 0.0)

                # 4. Decay state
                state = state * torch.exp(g_end)

                # 5. Update: h += k^T @ v_gated
                #    k_adj = k * alpha already has the gating
                k_adj = k[b_idx, tc, h_idx, :] * alpha[:, None]
                state = state + hl.dot(k_adj.T, diff, out_dtype=torch.float32)

        return h_out, v_out

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(k, w, u, g)
