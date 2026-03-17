from task import input_t, output_t

import torch
import helion
import helion.language as hl


# ------------------------------------------------------------
# Flip to True to autotune, False to use hardcoded winners
# ------------------------------------------------------------
AUTOTUNE = False


def _configs(v: int, k: int):
    cfgs = []
    for vb in [16, 32, 64, 128]:
        if vb > v or v % vb != 0:
            continue
        for kb in [16, 32, 64, 128]:
            if kb > k or k % kb != 0:
                continue
            for nw in [1, 2, 4, 8]:
                for ns in [1, 2, 3, 4]:
                    cfgs.append(helion.Config(block_sizes=[vb, kb], num_warps=nw, num_stages=ns))
    if v not in [16, 32, 64, 128]:
        for nw in [1, 2, 4, 8]:
            for ns in [1, 2, 3, 4]:
                cfgs.append(helion.Config(block_sizes=[v, k], num_warps=nw, num_stages=ns))
    return cfgs


# Hardcoded winners (autotuned)
BEST_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[16, 16], num_warps=4, num_stages=1),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[32, 64], num_warps=8, num_stages=2),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[32, 64], num_warps=8, num_stages=2),
    # Benchmark shapes (autotuned winners)
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[16, 16], num_warps=4, num_stages=1),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[32, 64], num_warps=8, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=1),
    # V=100/128 shapes — estimated, autotune later
    (3, 1024, 4, 100, 100): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=1),
    (4, 1024, 4, 128, 128): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=1),
    (2, 1536, 4, 128, 128): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=1),
    (4, 2048, 8, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=8, num_stages=1),
}

# Autotune search space
SEARCH_CONFIGS: dict[tuple, list[helion.Config]] = {
    (1, 64, 2, 64, 64): _configs(64, 64),
    (2, 128, 4, 64, 64): _configs(64, 64),
    (1, 256, 4, 64, 128): _configs(128, 64),
    (1, 64, 1, 64, 64): _configs(64, 64),
    (2, 512, 3, 64, 64): _configs(64, 64),
    (2, 1024, 3, 64, 64): _configs(64, 64),
    (3, 1024, 4, 100, 100): _configs(100, 100),
    (4, 1024, 4, 128, 128): _configs(128, 128),
    (2, 1536, 4, 128, 128): _configs(128, 128),
    (4, 2048, 8, 64, 64): _configs(64, 64),
}


def _make_kernel_static(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
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
        for flat_bh, rt, tile_v, tile_k in hl.tile(
            [BH, T, V, K], block_size=[1, C, None, None]
        ):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H

            beta_vals = beta[b_idx, rt, h_idx]
            g_vals = g[b_idx, rt, h_idx]
            A_tile = A[b_idx, rt, h_idx, :]

            # u = A @ (v * beta[:, None])
            v_tile = v[b_idx, rt, h_idx, tile_v]
            bv = v_tile * beta_vals[:, None]
            u_out[b_idx, rt, h_idx, tile_v] = hl.dot(A_tile, bv).to(v.dtype)

            # w = A @ (k * (beta * exp(g))[:, None])
            k_tile = k[b_idx, rt, h_idx, tile_k]
            kbg = k_tile * (beta_vals * torch.exp(g_vals))[:, None]
            w_out[b_idx, rt, h_idx, tile_k] = hl.dot(A_tile, kbg).to(k.dtype)

        return w_out, u_out

    return kernel


def _make_kernel_autotune(configs: list[helion.Config]):
    @helion.kernel(static_shapes=True, dot_precision="ieee", configs=configs, autotune_search_acf=True)
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
        for flat_bh, rt, tile_v, tile_k in hl.tile(
            [BH, T, V, K], block_size=[1, C, None, None]
        ):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H

            beta_vals = beta[b_idx, rt, h_idx]
            g_vals = g[b_idx, rt, h_idx]
            A_tile = A[b_idx, rt, h_idx, :]

            v_tile = v[b_idx, rt, h_idx, tile_v]
            bv = v_tile * beta_vals[:, None]
            u_out[b_idx, rt, h_idx, tile_v] = hl.dot(A_tile, bv).to(v.dtype)

            k_tile = k[b_idx, rt, h_idx, tile_k]
            kbg = k_tile * (beta_vals * torch.exp(g_vals))[:, None]
            w_out[b_idx, rt, h_idx, tile_k] = hl.dot(A_tile, kbg).to(k.dtype)

        return w_out, u_out

    return kernel


if AUTOTUNE:
    _KERNELS = {shape: _make_kernel_autotune(cfgs) for shape, cfgs in SEARCH_CONFIGS.items()}
else:
    _KERNELS = {shape: _make_kernel_static(cfg) for shape, cfg in BEST_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    return kernel(k, v, beta, A, g)
