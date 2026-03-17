from task import input_t, output_t

import torch
import helion
import helion.language as hl


@helion.kernel(config=helion.Config(block_sizes=[32, 64], num_warps=4, num_stages=3))
def causal_conv1d_kernel(
    x_pad: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    B = x_pad.size(0)
    D = x_pad.size(1)
    W = hl.specialize(w.size(1))
    N = x_pad.size(2) - W + 1

    y = torch.empty(B, D, N, dtype=x_pad.dtype, device=x_pad.device)

    for rb, rd, rs in hl.tile([B, D, N], block_size=[1, None, None]):
        bi = rb.begin
        acc = hl.zeros([rd, rs], dtype=torch.float32)
        acc = acc + b[rd].to(torch.float32)[:, None]
        for k in range(W):
            wk = w[rd, k].to(torch.float32)
            xk = hl.load(x_pad, [bi, rd, rs.index + k]).to(torch.float32)
            acc = acc + xk * wk[:, None]
        y[rb, rd, rs] = acc[None, :, :].to(y.dtype)

    return y


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]
    pad_zeros = torch.zeros(B, D, W - 1, dtype=x.dtype, device=x.device)
    padded = torch.cat([pad_zeros, x], dim=2)
    return causal_conv1d_kernel(padded, weight, bias)
