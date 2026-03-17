#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius

from task import input_t, output_t

import torch
import helion
import helion.language as hl


# Round 2: [1,64]+8w+4s timed out on server; keep round-1 for test shapes.
# Benchmark shapes (1,1536,2048,4), (1,2560,2048,4), (1,2560,4096,4) tuned on B200.
OPTIMIZED = helion.Config(block_sizes=[1, 32], num_warps=4, num_stages=2)
B200_BENCH_1536_2048 = helion.Config(block_sizes=[32, 64], num_warps=1, num_stages=2)
B200_BENCH_2560_2048 = helion.Config(block_sizes=[32, 32], num_warps=4, num_stages=1)
B200_BENCH_2560_4096 = helion.Config(block_sizes=[32, 64], num_warps=1, num_stages=2)
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    (1, 64, 64, 4): OPTIMIZED,
    (2, 128, 128, 4): OPTIMIZED,
    (1, 256, 256, 3): OPTIMIZED,
    (1, 128, 64, 8): OPTIMIZED,
    (4, 64, 128, 4): OPTIMIZED,
    (1, 1536, 2048, 4): B200_BENCH_1536_2048,
    (1, 2560, 2048, 4): B200_BENCH_2560_2048,
    (1, 2560, 4096, 4): B200_BENCH_2560_4096,
}


def _make_kernel(config: helion.Config | None):
    if config is None:
        decorator = helion.kernel(static_shapes=True, autotune_effort="quick")
    else:
        decorator = helion.kernel(static_shapes=True, config=config)

    @decorator
    def kernel(
        x_pad: torch.Tensor,  # (B, D, L) L = S + W - 1, causal left-padded
        w: torch.Tensor,      # (D, W)
        b: torch.Tensor,      # (D,)
    ) -> torch.Tensor:
        B = x_pad.size(0)
        D = x_pad.size(1)
        L = x_pad.size(2)
        W = hl.specialize(w.size(1))
        N = L - W + 1  # output length S

        y = torch.empty(B, D, N, dtype=x_pad.dtype, device=x_pad.device)

        for rb, rd, rs in hl.tile([B, D, N], block_size=[1, None, None]):
            bi = rb.begin
            acc = hl.zeros([rd, rs], dtype=torch.float32)
            for j in range(W):
                c = w[rd, j].to(torch.float32)
                x_val = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)
                acc = acc + x_val * c[:, None]
            acc = acc + b[rd].to(torch.float32)[:, None]
            y[rb, rd, rs] = acc[None, :, :].to(y.dtype)

        return y

    return kernel


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def get_autotune_kernel():
    """Kernel with autotune_effort='quick' for B200 tuning. Use data_to_kernel_args(data) then kernel.autotune(args)."""
    return _make_kernel(None)


def data_to_kernel_args(data: input_t):
    """Convert custom_kernel input to the raw kernel args (x_pad, w, b)."""
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    pad_zeros = torch.zeros(B, D, W - 1, dtype=x.dtype, device=x.device)
    padded = torch.cat([pad_zeros, x], dim=2)
    return (padded, weight, bias)


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    kernel = _KERNELS[(B, D, S, W)]
    pad_zeros = torch.zeros(B, D, W - 1, dtype=x.dtype, device=x.device)
    padded = torch.cat([pad_zeros, x], dim=2)
    return kernel(padded, weight, bias)
