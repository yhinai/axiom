#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Memory-bound elementwise kernel. H200 benchmarking favored smaller S tiles with
# lower warp counts on the official benchmark shapes.
_BENCH = helion.Config(block_sizes=[1, 512], num_warps=2, num_stages=1)
_DEFAULT = helion.Config(block_sizes=[1, 256], num_warps=4, num_stages=1)

SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 64, 4): _DEFAULT,
    (2, 128, 128, 4): _DEFAULT,
    (1, 256, 256, 3): _DEFAULT,
    (1, 128, 64, 8): _DEFAULT,
    (4, 64, 128, 4): _DEFAULT,
    # Benchmark shapes
    (1, 1536, 2048, 4): _BENCH,
    (1, 2560, 2048, 4): _BENCH,
    (1, 2560, 4096, 4): _BENCH,
}


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(
        x: torch.Tensor,      # (B, D, S)
        w: torch.Tensor,      # (D, W)
        b: torch.Tensor,      # (D,)
    ) -> torch.Tensor:
        B = x.size(0)
        D = x.size(1)
        S = x.size(2)
        W = hl.specialize(w.size(1))

        y = torch.empty(B, D, S, dtype=x.dtype, device=x.device)

        for rb, rd, rs in hl.tile([B, D, S], block_size=[1, None, None]):
            bi = rb.begin
            acc = hl.zeros([rd, rs], dtype=torch.float32)
            for j in range(W):
                src_idx = rs.index + j - (W - 1)
                safe_idx = src_idx.clamp(min=0)
                x_val = hl.load(x, [bi, rd, safe_idx]).to(torch.float32)
                valid = (src_idx >= 0).to(torch.float32)
                coeff = w[rd, j].to(torch.float32)
                acc = acc + x_val * coeff[:, None] * valid[None, :]
            acc = acc + b[rd].to(torch.float32)[:, None]
            y[rb, rd, rs] = acc[None, :, :]

        return y

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    kernel = _KERNELS[(B, D, S, W)]
    return kernel(x, weight, bias)
