from task import input_t, output_t

import torch
import triton
import triton.language as tl
import helion
import helion.language as hl


CHUNK_SIZE = 64
BLOCK_V = 32
BLOCK_K = 32

RECURRENT_BLOCK_V = 32
RECURRENT_NUM_WARPS = 4
RECURRENT_NUM_STAGES = 2

torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


@triton.jit
def _recurrent64_kernel(
    k_ptr,
    w_ptr,
    u_ptr,
    g_scale_ptr,
    g_last_ptr,
    h_ptr,
    v_ptr,
    stride_k_bh, stride_k_nt, stride_k_t, stride_k_k,
    stride_w_bh, stride_w_nt, stride_w_t, stride_w_k,
    stride_u_bh, stride_u_nt, stride_u_t, stride_u_v,
    stride_gs_bh, stride_gs_nt, stride_gs_t,
    stride_gl_bh, stride_gl_nt,
    stride_h_bh, stride_h_nt, stride_h_k, stride_h_v,
    stride_v_bh, stride_v_nt, stride_v_t, stride_v_v,
    NT,
    BLOCK_T: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_t = tl.arange(0, BLOCK_T)
    offs_k = tl.arange(0, 64)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    state = tl.zeros((64, BLOCK_V), dtype=tl.float32)

    for chunk_idx in tl.range(0, NT, num_stages=1, loop_unroll_factor=1, disable_licm=True):
        h_ptrs = (
            h_ptr
            + pid_bh * stride_h_bh
            + chunk_idx * stride_h_nt
            + offs_k[:, None] * stride_h_k
            + offs_v[None, :] * stride_h_v
        )
        tl.store(h_ptrs, state, mask=offs_v[None, :] < 64)

        w_ptrs = (
            w_ptr
            + pid_bh * stride_w_bh
            + chunk_idx * stride_w_nt
            + offs_t[:, None] * stride_w_t
            + offs_k[None, :] * stride_w_k
        )
        u_ptrs = (
            u_ptr
            + pid_bh * stride_u_bh
            + chunk_idx * stride_u_nt
            + offs_t[:, None] * stride_u_t
            + offs_v[None, :] * stride_u_v
        )
        g_scale_ptrs = g_scale_ptr + pid_bh * stride_gs_bh + chunk_idx * stride_gs_nt + offs_t * stride_gs_t
        v_ptrs = (
            v_ptr
            + pid_bh * stride_v_bh
            + chunk_idx * stride_v_nt
            + offs_t[:, None] * stride_v_t
            + offs_v[None, :] * stride_v_v
        )

        w = tl.load(w_ptrs)
        u = tl.load(u_ptrs, mask=offs_v[None, :] < 64, other=0.0)
        g_scale = tl.load(g_scale_ptrs)[:, None]
        g_last = tl.load(g_last_ptr + pid_bh * stride_gl_bh + chunk_idx * stride_gl_nt)

        proj = tl.zeros((BLOCK_T, BLOCK_V), dtype=tl.float32)
        for kk in tl.static_range(0, 64):
            w_col = tl.sum(tl.where((offs_k[None, :] == kk), w, 0.0), axis=1)
            state_row = tl.sum(tl.where((offs_k[:, None] == kk), state, 0.0), axis=0)
            proj += w_col[:, None] * state_row[None, :]
        v_chunk = u - proj
        tl.store(v_ptrs, v_chunk, mask=offs_v[None, :] < 64)

        v_gated = v_chunk * g_scale

        k_ptrs = (
            k_ptr
            + pid_bh * stride_k_bh
            + chunk_idx * stride_k_nt
            + offs_t[:, None] * stride_k_t
            + offs_k[None, :] * stride_k_k
        )
        k = tl.load(k_ptrs)
        delta = tl.zeros((64, BLOCK_V), dtype=tl.float32)
        for tt in tl.static_range(0, BLOCK_T):
            k_row = tl.sum(tl.where((offs_t[:, None] == tt), k, 0.0), axis=0)
            v_row = tl.sum(tl.where((offs_t[:, None] == tt), v_gated, 0.0), axis=0)
            delta += k_row[:, None] * v_row[None, :]
        state = state * g_last + delta


def _multi_chunk_state_pass_recurrent64(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor) -> output_t:
    B, T, H, K = k.shape
    V = u.shape[-1]
    NT = T // CHUNK_SIZE
    BH = B * H

    k_chunks = k.permute(0, 2, 1, 3).contiguous().view(BH, NT, CHUNK_SIZE, K)
    w_chunks = w.permute(0, 2, 1, 3).contiguous().view(BH, NT, CHUNK_SIZE, K)
    u_chunks = u.permute(0, 2, 1, 3).contiguous().view(BH, NT, CHUNK_SIZE, V)
    g_chunks = g.permute(0, 2, 1).contiguous().view(BH, NT, CHUNK_SIZE).float()
    g_last_exp = torch.exp(g_chunks[:, :, -1]).contiguous()
    g_scales = torch.exp(g_chunks[:, :, -1, None] - g_chunks).contiguous()

    h_chunks = torch.empty(BH, NT, K, V, dtype=torch.float32, device=k.device)
    v_chunks = torch.empty(BH, NT, CHUNK_SIZE, V, dtype=torch.float32, device=k.device)

    grid = (BH, triton.cdiv(V, RECURRENT_BLOCK_V))
    _recurrent64_kernel[grid](
        k_chunks,
        w_chunks,
        u_chunks,
        g_scales,
        g_last_exp,
        h_chunks,
        v_chunks,
        *k_chunks.stride(),
        *w_chunks.stride(),
        *u_chunks.stride(),
        *g_scales.stride(),
        *g_last_exp.stride(),
        *h_chunks.stride(),
        *v_chunks.stride(),
        NT=NT,
        BLOCK_T=CHUNK_SIZE,
        BLOCK_V=RECURRENT_BLOCK_V,
        num_warps=RECURRENT_NUM_WARPS,
        num_stages=RECURRENT_NUM_STAGES,
    )

    h = h_chunks.view(B, H, NT, K, V).permute(0, 2, 1, 3, 4)
    v_new = v_chunks.view(B, H, T, V).permute(0, 2, 1, 3)
    return h, v_new


@helion.kernel(
    static_shapes=True,
    dot_precision="ieee",
    config=helion.Config(
        block_sizes=[],
        num_warps=8,
        num_stages=3,
        pid_type="persistent_blocked",
        num_sm_multiplier=1,
        maxnreg=128,
    ),
)
def _single_chunk_state_pass(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    V = u.shape[-1]
    C = CHUNK_SIZE
    K = hl.specialize(K)
    V = hl.specialize(V)

    NT = T // C
    BH = B * H

    h_out = torch.empty(B, NT, H, K, V, dtype=torch.float32, device=k.device)
    v_out = torch.empty_like(u)

    for flat_bh, tv in hl.tile([BH, V], block_size=[1, 16]):
        b_idx = flat_bh.begin // H
        h_idx = flat_bh.begin % H
        state = hl.zeros([K, tv], dtype=torch.float32)

        for tc in hl.tile(T, block_size=C):
            chunk_idx = tc.begin // C
            t_end = tc.begin + C - 1
            h_out[b_idx, chunk_idx, h_idx, :, tv] = state

            w_chunk = w[b_idx, tc, h_idx, :].to(torch.float32)
            u_chunk = u[b_idx, tc, h_idx, tv].to(torch.float32)
            v_chunk = u_chunk - hl.dot(w_chunk, state, out_dtype=torch.float32)
            v_out[b_idx, tc, h_idx, tv] = v_chunk.to(v_out.dtype)

            g_chunk = g[b_idx, tc, h_idx].to(torch.float32)
            g_last = g[b_idx, t_end, h_idx].to(torch.float32)
            v_gated = v_chunk * torch.exp(g_last - g_chunk)[:, None]

            state = state * torch.exp(g_last)
            k_chunk = k[b_idx, tc, h_idx, :].to(torch.float32)
            state = state + hl.dot(k_chunk.T, v_gated, out_dtype=torch.float32)

    return h_out, v_out


def _multi_chunk_state_pass(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor) -> output_t:
    B, T, H, K = k.shape
    V = u.shape[-1]
    BT = CHUNK_SIZE
    NT = T // BT
    BH = B * H

    k_bh = k.permute(0, 2, 1, 3).reshape(BH, T, K)
    w_bh = w.permute(0, 2, 1, 3).reshape(BH, T, K)
    u_bh = u.permute(0, 2, 1, 3).reshape(BH, T, V)
    g_bh = g.permute(0, 2, 1).reshape(BH, T)

    h_bh = torch.empty(BH, NT, K, V, dtype=torch.float32, device=k.device)
    v_bh = torch.empty(BH, T, V, dtype=torch.float32, device=k.device)

    for flat_bh in range(BH):
        h_state = torch.zeros(K, V, dtype=torch.float32, device=k.device)

        for c in range(NT):
            cs = c * BT
            ce = cs + BT

            h_bh[flat_bh, c] = h_state

            w_chunk = w_bh[flat_bh, cs:ce]
            u_chunk = u_bh[flat_bh, cs:ce]
            v_chunk = u_chunk - torch.mm(w_chunk, h_state)
            v_bh[flat_bh, cs:ce] = v_chunk

            g_chunk = g_bh[flat_bh, cs:ce]
            g_last = g_chunk[-1]
            v_gated = v_chunk * torch.exp(g_last - g_chunk)[:, None]

            h_state = h_state * torch.exp(g_last)
            k_chunk = k_bh[flat_bh, cs:ce]
            h_state = h_state + torch.mm(k_chunk.T, v_gated)

    h = h_bh.reshape(B, H, NT, K, V).permute(0, 2, 1, 3, 4)
    v_new = v_bh.reshape(B, H, T, V).permute(0, 2, 1, 3)
    return h, v_new


if hasattr(torch, "compile"):
    _multi_chunk_state_pass = torch.compile(
        _multi_chunk_state_pass,
        fullgraph=False,
        dynamic=False,
        mode="default",
    )


@triton.jit
def _project_kernel(
    w_ptr,
    u_ptr,
    state_ptr,
    out_ptr,
    stride_w_bh, stride_w_t, stride_w_k,
    stride_u_bh, stride_u_t, stride_u_v,
    stride_s_bh, stride_s_k, stride_s_v,
    stride_o_bh, stride_o_t, stride_o_v,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_t = tl.arange(0, BLOCK_T)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    acc = tl.zeros((BLOCK_T, BLOCK_V), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        w_ptrs = w_ptr + pid_bh * stride_w_bh + offs_t[:, None] * stride_w_t + offs_k[None, :] * stride_w_k
        s_ptrs = state_ptr + pid_bh * stride_s_bh + offs_k[:, None] * stride_s_k + offs_v[None, :] * stride_s_v

        w = tl.load(w_ptrs, mask=offs_k[None, :] < K, other=0.0)
        s = tl.load(s_ptrs, mask=(offs_k[:, None] < K) & (offs_v[None, :] < V), other=0.0)
        acc += tl.dot(w, s, input_precision="ieee")

    u_ptrs = u_ptr + pid_bh * stride_u_bh + offs_t[:, None] * stride_u_t + offs_v[None, :] * stride_u_v
    o_ptrs = out_ptr + pid_bh * stride_o_bh + offs_t[:, None] * stride_o_t + offs_v[None, :] * stride_o_v
    u = tl.load(u_ptrs, mask=offs_v[None, :] < V, other=0.0)
    out = u - acc
    tl.store(o_ptrs, out, mask=offs_v[None, :] < V)


@triton.jit
def _update_kernel(
    k_ptr,
    vg_ptr,
    out_ptr,
    stride_k_bh, stride_k_t, stride_k_k,
    stride_vg_bh, stride_vg_t, stride_vg_v,
    stride_o_bh, stride_o_k, stride_o_v,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_v = tl.program_id(2)

    offs_t = tl.arange(0, BLOCK_T)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)

    k_ptrs = k_ptr + pid_bh * stride_k_bh + offs_t[:, None] * stride_k_t + offs_k[None, :] * stride_k_k
    vg_ptrs = vg_ptr + pid_bh * stride_vg_bh + offs_t[:, None] * stride_vg_t + offs_v[None, :] * stride_vg_v

    k = tl.load(k_ptrs, mask=offs_k[None, :] < K, other=0.0)
    vg = tl.load(vg_ptrs, mask=offs_v[None, :] < V, other=0.0)
    acc = tl.dot(tl.trans(k), vg, input_precision="ieee")

    o_ptrs = out_ptr + pid_bh * stride_o_bh + offs_k[:, None] * stride_o_k + offs_v[None, :] * stride_o_v
    tl.store(o_ptrs, acc, mask=(offs_k[:, None] < K) & (offs_v[None, :] < V))


def _project_chunk_triton(w_chunk: torch.Tensor, u_chunk: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    BH, _, K = w_chunk.shape
    V = u_chunk.shape[-1]
    out = torch.empty_like(u_chunk)
    grid = (BH, triton.cdiv(V, BLOCK_V))
    _project_kernel[grid](
        w_chunk,
        u_chunk,
        state,
        out,
        *w_chunk.stride(),
        *u_chunk.stride(),
        *state.stride(),
        *out.stride(),
        K,
        V,
        BLOCK_T=CHUNK_SIZE,
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
        num_warps=8,
        num_stages=3,
    )
    return out


def _update_chunk_triton(k_chunk: torch.Tensor, v_gated: torch.Tensor) -> torch.Tensor:
    BH, _, K = k_chunk.shape
    V = v_gated.shape[-1]
    out = torch.empty(BH, K, V, dtype=torch.float32, device=k_chunk.device)
    grid = (BH, triton.cdiv(K, BLOCK_K), triton.cdiv(V, BLOCK_V))
    _update_kernel[grid](
        k_chunk,
        v_gated,
        out,
        *k_chunk.stride(),
        *v_gated.stride(),
        *out.stride(),
        K,
        V,
        BLOCK_T=CHUNK_SIZE,
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
        num_warps=4,
        num_stages=2,
    )
    return out


def _multi_chunk_state_pass_triton(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor) -> output_t:
    B, T, H, K = k.shape
    V = u.shape[-1]
    NT = T // CHUNK_SIZE
    BH = B * H

    k_chunks = k.permute(0, 2, 1, 3).contiguous().view(BH, NT, CHUNK_SIZE, K).float()
    w_chunks = w.permute(0, 2, 1, 3).contiguous().view(BH, NT, CHUNK_SIZE, K)
    u_chunks = u.permute(0, 2, 1, 3).contiguous().view(BH, NT, CHUNK_SIZE, V)
    g_chunks = g.permute(0, 2, 1).contiguous().view(BH, NT, CHUNK_SIZE).float()
    g_last = g_chunks[:, :, -1]
    g_last_exp = torch.exp(g_last)
    g_scales = torch.exp(g_last[:, :, None] - g_chunks)

    h_bh = torch.empty(BH, NT, K, V, dtype=torch.float32, device=k.device)
    v_chunks = torch.empty(BH, NT, CHUNK_SIZE, V, dtype=torch.float32, device=k.device)
    state = torch.zeros(BH, K, V, dtype=torch.float32, device=k.device)

    for chunk_idx in range(NT):
        h_bh[:, chunk_idx] = state

        v_chunk = _project_chunk_triton(w_chunks[:, chunk_idx], u_chunks[:, chunk_idx], state)
        v_chunks[:, chunk_idx] = v_chunk

        v_gated = v_chunk * g_scales[:, chunk_idx, :, None]
        delta = _update_chunk_triton(k_chunks[:, chunk_idx], v_gated)
        state = state * g_last_exp[:, chunk_idx][:, None, None] + delta

    h = h_bh.view(B, H, NT, K, V).permute(0, 2, 1, 3, 4)
    v_new = v_chunks.view(B, H, T, V).permute(0, 2, 1, 3)
    return h, v_new


def _multi_chunk_state_pass_baddbmm(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor) -> output_t:
    B, T, H, K = k.shape
    V = u.shape[-1]
    NT = T // CHUNK_SIZE
    BH = B * H

    k_chunks = k.permute(0, 2, 1, 3).contiguous().view(BH, NT, CHUNK_SIZE, K).float()
    k_t_chunks = k_chunks.transpose(-1, -2).contiguous()
    w_chunks = w.permute(0, 2, 1, 3).contiguous().view(BH, NT, CHUNK_SIZE, K).float()
    u_chunks = u.permute(0, 2, 1, 3).contiguous().view(BH, NT, CHUNK_SIZE, V).float()
    g_chunks = g.permute(0, 2, 1).contiguous().view(BH, NT, CHUNK_SIZE).float()

    g_last = g_chunks[:, :, -1]
    g_last_exp = torch.exp(g_last)
    g_scales = torch.exp(g_last[:, :, None] - g_chunks)

    h_chunks = torch.empty(BH, NT, K, V, dtype=torch.float32, device=k.device)
    v_chunks = torch.empty(BH, NT, CHUNK_SIZE, V, dtype=torch.float32, device=k.device)
    state = torch.zeros(BH, K, V, dtype=torch.float32, device=k.device)

    for chunk_idx in range(NT):
        h_chunks[:, chunk_idx] = state
        v_chunk = torch.baddbmm(u_chunks[:, chunk_idx], w_chunks[:, chunk_idx], state, beta=1.0, alpha=-1.0)
        v_chunks[:, chunk_idx] = v_chunk

        v_gated = v_chunk * g_scales[:, chunk_idx, :, None]
        delta = torch.bmm(k_t_chunks[:, chunk_idx], v_gated)
        state = state * g_last_exp[:, chunk_idx][:, None, None] + delta

    h = h_chunks.view(B, H, NT, K, V).permute(0, 2, 1, 3, 4)
    v_new = v_chunks.view(B, H, T, V).permute(0, 2, 1, 3)
    return h, v_new


if hasattr(torch, "compile"):
    _multi_chunk_state_pass_baddbmm = torch.compile(
        _multi_chunk_state_pass_baddbmm,
        fullgraph=False,
        dynamic=False,
        mode="default",
    )


@torch.no_grad()
def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    _, T, _, K = k.shape
    V = u.shape[-1]

    if K == 64 and V == 64:
        if T == CHUNK_SIZE:
            return _single_chunk_state_pass(k, w, u, g)
        return _multi_chunk_state_pass_baddbmm(k, w, u, g)

    if T == CHUNK_SIZE:
        return _single_chunk_state_pass(k, w, u, g)

    return _multi_chunk_state_pass(k, w, u, g)
