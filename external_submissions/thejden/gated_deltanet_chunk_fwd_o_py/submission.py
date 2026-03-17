#!POPCORN leaderboard gated_deltanet_chunk_fwd_o
#!POPCORN gpu B200_Nebius
from __future__ import annotations

import torch
import helion
import helion.language as hl
from task import input_t, output_t

CHUNK_SIZE = 64

# Per-shape configs: map (B, T, H, K, V) to optimized helion.Config objects.
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Benchmark shapes (autotuned)
    (1, 64, 1, 64, 64): helion.Config(advanced_controls_file='/opt/booster_pack/chunk_fwd_o_0.acf', block_sizes=[], indexing=['tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], load_eviction_policies=['', '', 'last', 'last', '', 'first', ''], num_stages=1, num_warps=8, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None]),
    (2, 512, 3, 64, 64): helion.Config(advanced_controls_file='/opt/booster_pack/chunk_fwd_o_0.acf', block_sizes=[], indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], load_eviction_policies=['', '', 'last', '', '', '', 'last'], num_stages=1, num_warps=16, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None]),
    (2, 1024, 3, 64, 64): helion.Config(advanced_controls_file='/opt/booster_pack/chunk_fwd_o_0.acf', block_sizes=[], indexing=['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], load_eviction_policies=['', '', 'last', '', '', '', ''], num_stages=1, num_warps=8, pid_type='flat', range_flattens=[None], range_multi_buffers=[None], range_num_stages=[0], range_unroll_factors=[0], range_warp_specializes=[None]),
}

# Causal mask — allocated once at first use
_CAUSAL: torch.Tensor | None = None


@helion.kernel(autotune_effort="none", dot_precision="ieee")
def gated_deltanet_chunk_fwd_o_kernel(
    q: torch.Tensor,       # [BNH, C, K]
    k: torch.Tensor,       # [BNH, C, K]
    v_new: torch.Tensor,   # [BNH, C, V]
    h: torch.Tensor,       # [BNH, K, V]
    g: torch.Tensor,       # [BNH, C]
    causal: torch.Tensor,  # [C, C]
    scale: hl.constexpr,
) -> torch.Tensor:
    BNH = q.size(0)
    C = hl.specialize(q.size(1))
    K = hl.specialize(q.size(2))
    V = hl.specialize(v_new.size(2))

    out = torch.empty([BNH, C, V], dtype=q.dtype, device=q.device)

    for bnh in hl.grid(BNH):
        g_vec = g[bnh, :]                               # [C]
        q_b = q[bnh, :, :]                              # [C, K]

        # inter = (q @ h) * exp(g)
        inter = torch.mm(q_b, h[bnh, :, :]) * torch.exp(g_vec)[:, None]  # [C, V]

        # g_diff[i,j] = g[i] - g[j], zeroed in upper triangle to avoid exp overflow
        g_diff = (g_vec[:, None] - g_vec[None, :]) * causal[:, :]  # [C, C]

        # qk = (q @ k^T) * exp(g_diff) * causal
        qk = torch.mm(q_b, k[bnh, :, :].permute(1, 0)) * torch.exp(g_diff) * causal[:, :]

        # intra = qk @ v
        intra = torch.mm(qk, v_new[bnh, :, :])         # [C, V]

        out[bnh, :, :] = (inter + intra) * scale

    return out


def _reshape_to_chunks(q, k, v_new, h, g):
    """Reshape inputs from [B, T, H, *] to [B*NT*H, C, *] for the kernel."""
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    C = CHUNK_SIZE
    NT = T // C
    BNH = B * NT * H

    q_c = q.reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4).reshape(BNH, C, K)
    k_c = k.reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4).reshape(BNH, C, K)
    v_c = v_new.reshape(B, NT, C, H, V).permute(0, 1, 3, 2, 4).reshape(BNH, C, V)
    h_c = h.reshape(BNH, K, V)
    g_c = g.reshape(B, NT, C, H).permute(0, 1, 3, 2).reshape(BNH, C)

    return q_c, k_c, v_c, h_c, g_c


# Pre-compile and warm up a runner for each shape
_RUNNERS: dict[tuple, object] = {}

for (_B, _T, _H, _K, _V), _cfg in SHAPE_CONFIGS.items():
    _C = CHUNK_SIZE
    _NT = _T // _C
    _BNH = _B * _NT * _H
    _scale = _K ** -0.5

    _eq = torch.empty(_BNH, _C, _K, dtype=torch.float32, device="cuda")
    _ek = torch.empty(_BNH, _C, _K, dtype=torch.float32, device="cuda")
    _ev = torch.empty(_BNH, _C, _V, dtype=torch.float32, device="cuda")
    _eh = torch.empty(_BNH, _K, _V, dtype=torch.float32, device="cuda")
    _eg = torch.empty(_BNH, _C, dtype=torch.float32, device="cuda")
    if _CAUSAL is None:
        _CAUSAL = torch.tril(torch.ones(_C, _C, dtype=torch.float32, device="cuda"))
    _ec = _CAUSAL

    _bound = gated_deltanet_chunk_fwd_o_kernel.bind((_eq, _ek, _ev, _eh, _eg, _ec, _scale))
    _runner = _bound.compile_config(_cfg)
    _runner(_eq, _ek, _ev, _eh, _eg, _ec, _scale)
    _RUNNERS[(_B, _T, _H, _K, _V)] = _runner

if SHAPE_CONFIGS:
    torch.cuda.synchronize()


def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    C = CHUNK_SIZE
    NT = T // C
    scale = K ** -0.5

    q_c, k_c, v_c, h_c, g_c = _reshape_to_chunks(q, k, v_new, h, g)

    global _CAUSAL
    if _CAUSAL is None:
        _CAUSAL = torch.tril(torch.ones(C, C, dtype=torch.float32, device=q.device))

    key = (B, T, H, K, V)
    if key in _RUNNERS:
        o_c = _RUNNERS[key](q_c, k_c, v_c, h_c, g_c, _CAUSAL, scale)
    else:
        o_c = gated_deltanet_chunk_fwd_o_kernel(q_c, k_c, v_c, h_c, g_c, _CAUSAL, scale)

    return o_c.reshape(B, NT, H, C, V).permute(0, 1, 3, 2, 4).reshape(B, T, H, V)
