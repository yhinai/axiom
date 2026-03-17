from task import input_t, output_t

import torch
import triton
import triton.language as tl


CHUNK_SIZE = 64
BLOCK_N = 32
BLOCK_TK = 32

torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


def _project_kv_chunked_merged64(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
) -> output_t:
    B, T, H, K = k.shape
    NC = T // CHUNK_SIZE
    batch = B * NC * H

    k_chunks = k.view(B, NC, CHUNK_SIZE, H, K).permute(0, 1, 3, 2, 4).float()
    v_chunks = v.view(B, NC, CHUNK_SIZE, H, CHUNK_SIZE).permute(0, 1, 3, 2, 4).float()
    beta_chunks = beta.view(B, NC, CHUNK_SIZE, H).permute(0, 1, 3, 2).float()
    g_chunks = g.view(B, NC, CHUNK_SIZE, H).permute(0, 1, 3, 2).float()
    A_chunks = A.view(B, NC, CHUNK_SIZE, H, CHUNK_SIZE).permute(0, 1, 3, 2, 4).float()

    A_mat = A_chunks.reshape(batch, CHUNK_SIZE, CHUNK_SIZE)
    beta_scale = beta_chunks.reshape(batch, CHUNK_SIZE, 1)
    gate_scale = torch.exp(g_chunks).reshape(batch, CHUNK_SIZE, 1)

    u_in = v_chunks.reshape(batch, CHUNK_SIZE, CHUNK_SIZE) * beta_scale
    w_in = k_chunks.reshape(batch, CHUNK_SIZE, CHUNK_SIZE) * beta_scale * gate_scale
    uw_out = torch.matmul(A_mat, torch.cat((u_in, w_in), dim=-1))
    u_out, w_out = uw_out.split(CHUNK_SIZE, dim=-1)

    u = u_out.view(B, NC, H, CHUNK_SIZE, CHUNK_SIZE).permute(0, 1, 3, 2, 4).reshape(B, T, H, CHUNK_SIZE)
    w = w_out.view(B, NC, H, CHUNK_SIZE, CHUNK_SIZE).permute(0, 1, 3, 2, 4).reshape(B, T, H, CHUNK_SIZE)
    return w.to(k.dtype), u.to(v.dtype)


def _project_kv_chunked_torch(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
) -> output_t:
    B, T, H, K = k.shape
    V = v.shape[-1]
    NC = T // CHUNK_SIZE

    k_chunks = k.view(B, NC, CHUNK_SIZE, H, K).permute(0, 1, 3, 2, 4).float()
    v_chunks = v.view(B, NC, CHUNK_SIZE, H, V).permute(0, 1, 3, 2, 4).float()
    beta_chunks = beta.view(B, NC, CHUNK_SIZE, H).permute(0, 1, 3, 2).float()
    g_chunks = g.view(B, NC, CHUNK_SIZE, H).permute(0, 1, 3, 2).float()
    A_chunks = A.view(B, NC, CHUNK_SIZE, H, CHUNK_SIZE).permute(0, 1, 3, 2, 4).float()

    batch = B * NC * H
    A_mat = A_chunks.reshape(batch, CHUNK_SIZE, CHUNK_SIZE)
    beta_scale = beta_chunks.reshape(batch, CHUNK_SIZE, 1)
    gate_scale = torch.exp(g_chunks).reshape(batch, CHUNK_SIZE, 1)

    u_in = v_chunks.reshape(batch, CHUNK_SIZE, V) * beta_scale
    w_in = k_chunks.reshape(batch, CHUNK_SIZE, K) * beta_scale * gate_scale

    u_out = torch.matmul(A_mat, u_in)
    w_out = torch.matmul(A_mat, w_in)

    u = u_out.view(B, NC, H, CHUNK_SIZE, V).permute(0, 1, 3, 2, 4).reshape(B, T, H, V)
    w = w_out.view(B, NC, H, CHUNK_SIZE, K).permute(0, 1, 3, 2, 4).reshape(B, T, H, K)
    return w.to(k.dtype), u.to(v.dtype)


_project_kv_chunked_fast64_default = _project_kv_chunked_merged64
if hasattr(torch, "compile"):
    _project_kv_chunked_fast64_default = torch.compile(
        _project_kv_chunked_fast64_default,
        fullgraph=False,
        dynamic=False,
        mode="default",
    )

_project_kv_chunked_fast64_ro = _project_kv_chunked_merged64
if hasattr(torch, "compile"):
    _project_kv_chunked_fast64_ro = torch.compile(
        _project_kv_chunked_fast64_ro,
        fullgraph=False,
        dynamic=False,
        mode="reduce-overhead",
    )


@triton.jit
def _project_kv_fused64_kernel(
    A_ptr,
    k_ptr,
    v_ptr,
    beta_ptr,
    g_ptr,
    w_ptr,
    u_ptr,
    stride_A_b,
    stride_A_m,
    stride_A_k,
    stride_k_b,
    stride_k_t,
    stride_k_n,
    stride_v_b,
    stride_v_t,
    stride_v_n,
    stride_beta_b,
    stride_beta_t,
    stride_g_b,
    stride_g_t,
    stride_w_b,
    stride_w_t,
    stride_w_n,
    stride_u_b,
    stride_u_t,
    stride_u_n,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_T)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < BLOCK_T

    acc_w = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, BLOCK_T, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        A_ptrs = A_ptr + pid_b * stride_A_b + offs_m[:, None] * stride_A_m + offs_k[None, :] * stride_A_k
        k_ptrs = k_ptr + pid_b * stride_k_b + offs_k[:, None] * stride_k_t + offs_n[None, :] * stride_k_n
        v_ptrs = v_ptr + pid_b * stride_v_b + offs_k[:, None] * stride_v_t + offs_n[None, :] * stride_v_n
        beta_ptrs = beta_ptr + pid_b * stride_beta_b + offs_k * stride_beta_t
        g_ptrs = g_ptr + pid_b * stride_g_b + offs_k * stride_g_t

        A_tile = tl.load(A_ptrs)
        k_tile = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        v_tile = tl.load(v_ptrs, mask=mask_n[None, :], other=0.0)
        beta_tile = tl.load(beta_ptrs)
        g_tile = tl.load(g_ptrs)

        u_in = v_tile * beta_tile[:, None]
        w_in = k_tile * (beta_tile * tl.exp(g_tile))[:, None]

        acc_u += tl.dot(A_tile, u_in, input_precision="ieee")
        acc_w += tl.dot(A_tile, w_in, input_precision="ieee")

    w_ptrs = w_ptr + pid_b * stride_w_b + offs_m[:, None] * stride_w_t + offs_n[None, :] * stride_w_n
    u_ptrs = u_ptr + pid_b * stride_u_b + offs_m[:, None] * stride_u_t + offs_n[None, :] * stride_u_n
    tl.store(w_ptrs, acc_w, mask=mask_n[None, :])
    tl.store(u_ptrs, acc_u, mask=mask_n[None, :])


def _project_kv_chunked_triton64(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
) -> output_t:
    B, T, H, K = k.shape
    NC = T // CHUNK_SIZE
    batch = B * NC * H

    k_chunks = k.view(B, NC, CHUNK_SIZE, H, K).permute(0, 1, 3, 2, 4).contiguous().reshape(batch, CHUNK_SIZE, CHUNK_SIZE)
    v_chunks = v.view(B, NC, CHUNK_SIZE, H, CHUNK_SIZE).permute(0, 1, 3, 2, 4).contiguous().reshape(batch, CHUNK_SIZE, CHUNK_SIZE)
    beta_chunks = beta.view(B, NC, CHUNK_SIZE, H).permute(0, 1, 3, 2).contiguous().reshape(batch, CHUNK_SIZE)
    g_chunks = g.view(B, NC, CHUNK_SIZE, H).permute(0, 1, 3, 2).contiguous().reshape(batch, CHUNK_SIZE)
    A_chunks = A.view(B, NC, CHUNK_SIZE, H, CHUNK_SIZE).permute(0, 1, 3, 2, 4).contiguous().reshape(batch, CHUNK_SIZE, CHUNK_SIZE)

    w_out = torch.empty_like(k_chunks)
    u_out = torch.empty_like(v_chunks)
    grid = (batch, triton.cdiv(CHUNK_SIZE, BLOCK_N))

    _project_kv_fused64_kernel[grid](
        A_chunks,
        k_chunks,
        v_chunks,
        beta_chunks,
        g_chunks,
        w_out,
        u_out,
        *A_chunks.stride(),
        *k_chunks.stride(),
        *v_chunks.stride(),
        *beta_chunks.stride(),
        *g_chunks.stride(),
        *w_out.stride(),
        *u_out.stride(),
        BLOCK_T=CHUNK_SIZE,
        BLOCK_K=BLOCK_TK,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    w = w_out.view(B, NC, H, CHUNK_SIZE, CHUNK_SIZE).permute(0, 1, 3, 2, 4).reshape(B, T, H, CHUNK_SIZE)
    u = u_out.view(B, NC, H, CHUNK_SIZE, CHUNK_SIZE).permute(0, 1, 3, 2, 4).reshape(B, T, H, CHUNK_SIZE)
    return w.to(k.dtype), u.to(v.dtype)


@torch.no_grad()
def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    _, T, H, K = k.shape
    V = v.shape[-1]

    if K == 64 and V == 64:
        if H == 3 and T >= 512:
            return _project_kv_chunked_fast64_ro(k, v, beta, A, g)
        return _project_kv_chunked_fast64_default(k, v, beta, A, g)

    return _project_kv_chunked_torch(k, v, beta, A, g)
