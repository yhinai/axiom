from task import input_t, output_t

import torch
import helion
import helion.language as hl


# ------------------------------------------------------------
# Flip to True to autotune, False to use hardcoded winners
# ------------------------------------------------------------
AUTOTUNE = False


def _configs(v: int):
    cfgs = []
    for vb in [16, 32, 64, 128]:
        if vb > v or v % vb != 0:
            continue
        for nw in [1, 2, 4, 8]:
            for ns in [1, 2, 3, 4]:
                cfgs.append(helion.Config(block_sizes=[vb], num_warps=nw, num_stages=ns))
    if v not in [16, 32, 64, 128]:
        for nw in [1, 2, 4, 8]:
            for ns in [1, 2, 3, 4]:
                cfgs.append(helion.Config(block_sizes=[v], num_warps=nw, num_stages=ns))
    return cfgs


# Hardcoded winners (autotuned)
BEST_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[16], num_warps=4, num_stages=1),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[32], num_warps=8, num_stages=1),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[32], num_warps=8, num_stages=1),
    # Benchmark shapes (autotuned winners)
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[16], num_warps=4, num_stages=1),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[32], num_warps=8, num_stages=1),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[64], num_warps=8, num_stages=1),
    # V=100/128 shapes — need autotuning still
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[32], num_warps=8, num_stages=1),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[64], num_warps=8, num_stages=1),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[64], num_warps=8, num_stages=1),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[64], num_warps=8, num_stages=1),
}

# Autotune search space
SEARCH_CONFIGS: dict[tuple, list[helion.Config]] = {
    (1, 64, 2, 64, 64): _configs(64),
    (2, 128, 4, 64, 64): _configs(64),
    (1, 256, 4, 64, 128): _configs(128),
    (1, 64, 1, 64, 64): _configs(64),
    (2, 512, 3, 64, 64): _configs(64),
    (2, 1024, 3, 64, 64): _configs(64),
    (3, 1024, 4, 100, 100): _configs(100),
    (4, 1024, 4, 128, 128): _configs(128),
    (2, 1536, 4, 128, 128): _configs(128),
    (4, 2048, 8, 64, 64): _configs(64),
}


def _make_kernel_static(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
        g: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:

        B, T, H, K = q.shape
        V = v.shape[-1]

        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)

        out = torch.empty_like(v)

        BH = B * H

        for flat_bh, tile_t, tile_v in hl.tile(
            [BH, T, V],
            block_size=[1, C, None]
        ):

            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            c_idx = tile_t.begin // C

            g_vals = g[b_idx, tile_t, h_idx]

            q_tile = q[b_idx, tile_t, h_idx, :]
            k_tile = k[b_idx, tile_t, h_idx, :]

            # QK attention
            qk = hl.dot(q_tile, k_tile.T)

            g_diff = g_vals[:, None] - g_vals[None, :]

            idx = hl.arange(tile_t.block_size)
            causal_mask = idx[:, None] >= idx[None, :]

            attn = torch.where(
                causal_mask,
                qk * torch.exp(g_diff),
                0.0
            )

            v_slice = v[b_idx, tile_t, h_idx, tile_v]

            local_out = hl.dot(
                attn.to(v.dtype),
                v_slice
            )

            # Inter-chunk
            q_s = q_tile * torch.exp(g_vals)[:, None]

            h_slice = h[b_idx, c_idx, h_idx, :, tile_v]

            global_out = hl.dot(q_s, h_slice)

            out[b_idx, tile_t, h_idx, tile_v] = (
                (global_out + local_out) * scale
            ).to(out.dtype)

        return out

    return kernel


def _make_kernel_autotune(configs: list[helion.Config]):
    @helion.kernel(static_shapes=True, dot_precision="ieee", configs=configs, autotune_search_acf=True)
    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
        g: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:

        B, T, H, K = q.shape
        V = v.shape[-1]

        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)

        out = torch.empty_like(v)

        BH = B * H

        for flat_bh, tile_t, tile_v in hl.tile(
            [BH, T, V],
            block_size=[1, C, None]
        ):

            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            c_idx = tile_t.begin // C

            g_vals = g[b_idx, tile_t, h_idx]

            q_tile = q[b_idx, tile_t, h_idx, :]
            k_tile = k[b_idx, tile_t, h_idx, :]

            qk = hl.dot(q_tile, k_tile.T)

            g_diff = g_vals[:, None] - g_vals[None, :]

            idx = hl.arange(tile_t.block_size)
            causal_mask = idx[:, None] >= idx[None, :]

            attn = torch.where(
                causal_mask,
                qk * torch.exp(g_diff),
                0.0
            )

            v_slice = v[b_idx, tile_t, h_idx, tile_v]

            local_out = hl.dot(
                attn.to(v.dtype),
                v_slice
            )

            q_s = q_tile * torch.exp(g_vals)[:, None]

            h_slice = h[b_idx, c_idx, h_idx, :, tile_v]

            global_out = hl.dot(q_s, h_slice)

            out[b_idx, tile_t, h_idx, tile_v] = (
                (global_out + local_out) * scale
            ).to(out.dtype)

        return out

    return kernel


# ------------------------------------------------------------
# Kernel cache: flip AUTOTUNE at top to switch
# ------------------------------------------------------------

if AUTOTUNE:
    _KERNELS = {shape: _make_kernel_autotune(cfgs) for shape, cfgs in SEARCH_CONFIGS.items()}
else:
    _KERNELS = {shape: _make_kernel_static(cfg) for shape, cfg in BEST_CONFIGS.items()}


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def custom_kernel(data: input_t) -> output_t:

    q, k, v_new, h, g = data

    B, T, H, K = q.shape
    V = v_new.shape[-1]

    scale = K ** -0.5

    kernel = _KERNELS[(B, T, H, K, V)]

    return kernel(q, k, v_new, h, g, scale)