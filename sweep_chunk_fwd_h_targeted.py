#!/usr/bin/env python3
"""
Targeted H200-safe sweep for gated_deltanet_chunk_fwd_h.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import helion
import helion.language as hl
import torch


_LOG2E = 1.4426950408889634
BENCHMARK_SHAPES = [
    (1, 64, 1, 64, 64, 31232),
    (2, 512, 3, 64, 64, 4052),
    (2, 1024, 3, 64, 64, 2146),
]


@dataclass(frozen=True)
class ShapeResult:
    shape: tuple[int, int, int, int, int]
    median_s: float


@dataclass(frozen=True)
class SweepResult:
    label: str
    config: helion.Config
    shape_results: tuple[ShapeResult, ...] | None
    error: str | None = None

    @property
    def score(self) -> float | None:
        if self.shape_results is None:
            return None
        return sum(item.median_s for item in self.shape_results) / len(self.shape_results)


def make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, dot_precision="tf32", config=config)
    def kernel(
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = u.shape[-1]
        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)

        NT = (T + C - 1) // C
        h_out = torch.empty(B, NT, H, K, V, dtype=k.dtype, device=k.device)
        v_out = torch.empty_like(u)
        BH = B * H

        for flat, tv in hl.tile([BH, V], block_size=[1, 8]):
            b_idx = flat.begin // H
            h_idx = flat.begin % H
            state = hl.zeros([K, tv], dtype=torch.float32)

            for tc in hl.tile(T, block_size=C):
                chunk_idx = tc.begin // C
                t_end = min(tc.begin + C, T) - 1
                h_out[b_idx, chunk_idx, h_idx, :, tv] = state.to(k.dtype)

                proj = hl.dot(w[b_idx, tc, h_idx, :], state, out_dtype=torch.float32)
                diff = u[b_idx, tc, h_idx, tv].to(torch.float32) - proj
                v_out[b_idx, tc, h_idx, tv] = diff.to(u.dtype)

                g_end = g[b_idx, t_end, h_idx].to(torch.float32)
                g_t = g[b_idx, tc, h_idx].to(torch.float32)
                valid = tc.index < T
                alpha = torch.where(valid, torch.exp2((g_end - g_t) * _LOG2E), 0.0)
                diff_gated = diff * alpha[:, None]

                state = state * torch.exp2(g_end * _LOG2E)
                state = hl.dot(k[b_idx, tc, h_idx, :].T, diff_gated, acc=state, out_dtype=torch.float32)

        return h_out, v_out

    return kernel


def benchmark_kernel(kernel_fn, args, warmup: int = 6, iters: int = 20) -> float:
    for _ in range(warmup):
        kernel_fn(*args)
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        kernel_fn(*args)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2]


def make_inputs(B: int, T: int, H: int, K: int, V: int, seed: int):
    torch.manual_seed(seed)
    device = "cuda"
    C = 64

    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device) / math.sqrt(K)
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    beta = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g_inc = -torch.abs(torch.randn(B, T, H, dtype=torch.float32, device=device))
    g = g_inc.cumsum(dim=1)
    g_cumsum = g.float().reshape(B, T // C, C, H).cumsum(dim=2).reshape(B, T, H)

    k_c = k.float().reshape(B, T // C, C, H, K).permute(0, 1, 3, 2, 4)
    g_c = g_cumsum.float().reshape(B, T // C, C, H).permute(0, 1, 3, 2)
    beta_c = beta.float().reshape(B, T // C, C, H).permute(0, 1, 3, 2)
    kkt = k_c @ k_c.transpose(-1, -2)
    g_diff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)
    strict_lower = torch.tril(torch.ones(C, C, device=device), diagonal=-1)
    A = kkt * beta_c.unsqueeze(-1) * torch.exp(g_diff) * strict_lower
    A = A.permute(0, 1, 3, 2, 4).reshape(B, T, H, C).to(torch.float32)

    A_mat = A.float().reshape(B, T // C, C, H, C).permute(0, 1, 3, 2, 4)
    eye = torch.eye(C, device=device).expand_as(A_mat)
    A = torch.linalg.solve_triangular(eye + A_mat, eye, upper=False)
    A = A.permute(0, 1, 3, 2, 4).reshape(B, T, H, C).contiguous()

    v_c = v.float().reshape(B, T // C, C, H, V).permute(0, 1, 3, 2, 4)
    beta_cc = beta.float().reshape(B, T // C, C, H).permute(0, 1, 3, 2)
    g_cc = g_cumsum.float().reshape(B, T // C, C, H).permute(0, 1, 3, 2)
    A_c = A.float().reshape(B, T // C, C, H, C).permute(0, 1, 3, 2, 4)
    u_c = A_c @ (v_c * beta_cc.unsqueeze(-1))
    w_c = A_c @ (k_c * (beta_cc * torch.exp(g_cc)).unsqueeze(-1))
    w = w_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, K).contiguous()
    u = u_c.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).contiguous()

    return k.contiguous(), w.contiguous(), u.contiguous(), g_cumsum.contiguous()


def test_config(label: str, config: helion.Config) -> SweepResult:
    try:
        shape_results: list[ShapeResult] = []
        for B, T, H, K, V, seed in BENCHMARK_SHAPES:
            kernel_fn = make_kernel(config)
            args = make_inputs(B, T, H, K, V, seed)
            kernel_fn(*args)
            torch.cuda.synchronize()
            median_s = benchmark_kernel(kernel_fn, args)
            shape_results.append(ShapeResult((B, T, H, K, V), median_s))
        return SweepResult(label=label, config=config, shape_results=tuple(shape_results))
    except Exception as exc:
        return SweepResult(label=label, config=config, shape_results=None, error=str(exc)[:180])


def make_config(
    *,
    indexing,
    num_warps,
    num_stages,
    l2_groupings,
    loop_orders,
    load_eviction_policies,
    range_flattens,
    range_multi_buffers,
    range_num_stages,
    range_unroll_factors,
) -> helion.Config:
    return helion.Config(
        block_sizes=[],
        indexing=indexing,
        num_warps=num_warps,
        num_stages=num_stages,
        l2_groupings=l2_groupings,
        loop_orders=loop_orders,
        load_eviction_policies=load_eviction_policies,
        pid_type="flat",
        range_flattens=range_flattens,
        range_multi_buffers=range_multi_buffers,
        range_num_stages=range_num_stages,
        range_unroll_factors=range_unroll_factors,
    )


def iter_configs():
    base = dict(
        indexing=["tensor_descriptor", "pointer", "tensor_descriptor", "tensor_descriptor", "pointer", "tensor_descriptor", "pointer"],
        num_warps=4,
        num_stages=3,
        l2_groupings=[1],
        loop_orders=[[1, 0]],
        load_eviction_policies=["first", "", "", "", ""],
        range_flattens=[None, True],
        range_multi_buffers=[None, False],
        range_num_stages=[0, 3],
        range_unroll_factors=[0, 0],
    )
    variants = {
        "baseline": {},
        "nw8": {"num_warps": 8},
        "nw16": {"num_warps": 16},
        "ns1": {"num_stages": 1},
        "ns2": {"num_stages": 2},
        "ns5": {"num_stages": 5},
        "l2g4": {"l2_groupings": [4]},
        "tv_flat": {"loop_orders": [[0, 1]]},
        "rns1": {"range_num_stages": [0, 1]},
        "rns2": {"range_num_stages": [0, 2]},
        "ruf1": {"range_unroll_factors": [0, 1]},
        "all_pointer": {"indexing": ["pointer"] * 7, "load_eviction_policies": ["", "", "", "", ""]},
        "pointer_heavy": {"indexing": ["pointer", "pointer", "tensor_descriptor", "tensor_descriptor", "pointer", "tensor_descriptor", "pointer"], "load_eviction_policies": ["", "", "", "", ""]},
        "tiny_like": {"num_warps": 16},
    }
    seen: set[str] = set()
    for label, overrides in variants.items():
        cfg = make_config(**(base | overrides))
        key = repr(cfg)
        if key not in seen:
            seen.add(key)
            yield label, cfg


def main() -> int:
    print("Shape set:", [shape[:5] for shape in BENCHMARK_SHAPES], flush=True)
    print(f"Device: {torch.cuda.get_device_name()}", flush=True)
    print("=" * 80, flush=True)

    results = [test_config(label, cfg) for label, cfg in iter_configs()]
    for result in results:
        if result.score is None:
            print(f"FAIL {result.label}: {result.error}", flush=True)
            continue
        shape_bits = ", ".join(f"{item.median_s * 1e6:.1f}us" for item in result.shape_results or ())
        print(f"{result.label}: avg={result.score * 1e6:.1f}us [{shape_bits}]", flush=True)

    good = [item for item in results if item.score is not None]
    good.sort(key=lambda item: item.score)

    print("\n=== Top Results ===", flush=True)
    for idx, result in enumerate(good[:10], start=1):
        shape_bits = ", ".join(f"{item.shape}:{item.median_s * 1e6:.1f}us" for item in result.shape_results or ())
        print(f"{idx:02d}. {result.label}: avg={result.score * 1e6:.1f}us {shape_bits}", flush=True)
        print(f"    {result.config!r}", flush=True)

    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
