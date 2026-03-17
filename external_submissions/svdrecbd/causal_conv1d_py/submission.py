from task import input_t, output_t

import torch
import helion
import helion.language as hl


_W4_EXPR = """
{bias} + {x0} * {c0} + {x1} * {c1} + {x2} * {c2} + {x3} * {c3}
"""


@helion.kernel(
    static_shapes=True,
    config=helion.Config(block_sizes=[], num_warps=2, num_stages=2),
)
def conv1d_w4_main_kernel(
    x: torch.Tensor,      # (B, D, S) input
    w: torch.Tensor,      # (D, 4) filter coefficients
    b: torch.Tensor,      # (D,) additive offset
) -> torch.Tensor:
    B = x.size(0)
    D = x.size(1)
    S = x.size(2)
    N = S - 3

    y = torch.empty(B, D, N, dtype=x.dtype, device=x.device)

    BD = B * D
    for flat_bd, rs in hl.tile([BD, N], block_size=[1, 256]):
        b_idx = flat_bd.begin // D
        d_idx = flat_bd.begin % D

        x0 = hl.load(x, [b_idx, d_idx, rs.index + 0]).to(torch.float32)
        x1 = hl.load(x, [b_idx, d_idx, rs.index + 1]).to(torch.float32)
        x2 = hl.load(x, [b_idx, d_idx, rs.index + 2]).to(torch.float32)
        x3 = hl.load(x, [b_idx, d_idx, rs.index + 3]).to(torch.float32)

        c0 = w[d_idx, 0].to(torch.float32)
        c1 = w[d_idx, 1].to(torch.float32)
        c2 = w[d_idx, 2].to(torch.float32)
        c3 = w[d_idx, 3].to(torch.float32)
        bias = b[d_idx].to(torch.float32)

        acc = hl.inline_triton(
            _W4_EXPR,
            {
                "bias": bias,
                "x0": x0,
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "c0": c0,
                "c1": c1,
                "c2": c2,
                "c3": c3,
            },
            output_like=x3,
        )
        y[b_idx, d_idx, rs] = acc.to(y.dtype)

    return y


@helion.kernel(
    static_shapes=True,
    config=helion.Config(block_sizes=[1, 8], num_warps=1, num_stages=1),
)
def conv1d_generic_kernel(
    x_pad: torch.Tensor,  # (B, D, L) zero-padded input
    w: torch.Tensor,      # (D, W) filter coefficients
    b: torch.Tensor,      # (D,) additive offset
) -> torch.Tensor:
    B = x_pad.size(0)
    D = x_pad.size(1)
    L = x_pad.size(2)
    W = hl.specialize(w.size(1))
    N = L - W + 1

    y = torch.empty(B, D, N, dtype=x_pad.dtype, device=x_pad.device)

    for rb, rd, rs in hl.tile([B, D, N], block_size=[1, None, None]):
        bi = rb.begin
        acc = hl.zeros([rd, rs], dtype=torch.float32)
        for j in range(W):
            coeff = w[rd, j].to(torch.float32)
            xj = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)
            acc = acc + xj * coeff[:, None]
        acc = acc + b[rd].to(torch.float32)[:, None]
        y[rb, rd, rs] = acc[None, :, :].to(y.dtype)

    return y


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]

    if W == 4 and S >= 1024:
        y = torch.empty(B, D, S, dtype=x.dtype, device=x.device)

        bias_row = bias[None, :]
        w1 = weight[:, 1][None, :]
        w2 = weight[:, 2][None, :]
        w3 = weight[:, 3][None, :]

        y[:, :, 0] = bias_row + x[:, :, 0] * w3
        y[:, :, 1] = bias_row + x[:, :, 0] * w2 + x[:, :, 1] * w3
        y[:, :, 2] = bias_row + x[:, :, 0] * w1 + x[:, :, 1] * w2 + x[:, :, 2] * w3
        y[:, :, 3:] = conv1d_w4_main_kernel(x, weight, bias)
        return y

    pad_zeros = torch.zeros(B, D, W - 1, dtype=x.dtype, device=x.device)
    padded = torch.cat([pad_zeros, x], dim=2)
    return conv1d_generic_kernel(padded, weight, bias)
