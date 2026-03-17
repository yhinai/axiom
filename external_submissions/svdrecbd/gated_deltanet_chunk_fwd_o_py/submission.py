from task import input_t, output_t

import torch


CHUNK_SIZE = 64


def _gated_chunk_attn(q: torch.Tensor, k: torch.Tensor, v_new: torch.Tensor, h: torch.Tensor, g: torch.Tensor) -> output_t:
    B, T, H, K = q.shape
    BT = CHUNK_SIZE
    V = v_new.shape[-1]
    NC = T // BT
    BH = B * H
    batch = BH * NC
    scale = K ** -0.5

    causal = torch.tril(torch.ones(BT, BT, device=q.device, dtype=torch.bool)).unsqueeze(0)

    q_chunks = q.permute(0, 2, 1, 3).contiguous().view(BH, NC, BT, K).float()
    k_chunks = k.permute(0, 2, 1, 3).contiguous().view(BH, NC, BT, K).float()
    v_chunks = v_new.permute(0, 2, 1, 3).contiguous().view(BH, NC, BT, V).float()
    g_chunks = g.permute(0, 2, 1).contiguous().view(BH, NC, BT).float()
    h_chunks = h.permute(0, 2, 1, 3, 4).contiguous().view(BH, NC, K, V).float()

    g_pos = torch.exp(g_chunks).unsqueeze(-1)
    g_neg = torch.exp(-g_chunks).unsqueeze(-1)

    q_g = (q_chunks * g_pos).view(batch, BT, K)
    k_g = (k_chunks * g_neg).view(batch, BT, K)
    v_mat = v_chunks.view(batch, BT, V)
    h_mat = h_chunks.view(batch, K, V)

    inter = torch.bmm(q_g, h_mat)
    attn = torch.bmm(q_g, k_g.transpose(1, 2))
    attn = attn.masked_fill(~causal, 0.0)
    out = torch.baddbmm(inter, attn, v_mat)

    return (out * scale).view(B, H, NC, BT, V).reshape(B, H, T, V).permute(0, 2, 1, 3).contiguous().to(v_new.dtype)


if hasattr(torch, "compile"):
    _gated_chunk_attn = torch.compile(_gated_chunk_attn, fullgraph=False, dynamic=False, mode="reduce-overhead")


@torch.no_grad()
def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    return _gated_chunk_attn(q, k, v_new, h, g)
