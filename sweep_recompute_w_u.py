#!/usr/bin/env python3
"""
Targeted H200-safe sweep for gated_deltanet_recompute_w_u.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import helion
import helion.language as hl
import torch


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
    @helion.kernel(static_shapes=True, dot_precision="ieee", config=config)
    def kernel(
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, K = k.shape
        V = v.shape[-1]
        C = hl.specialize(A.shape[-1])
        K = hl.specialize(K)
        V = hl.specialize(V)

        w_out = torch.empty_like(k)
        u_out = torch.empty_like(v)
        BH = B * H

        for flat_bh, rt in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H

            beta_vals = beta[b_idx, rt, h_idx].to(torch.float32)
            g_vals = g[b_idx, rt, h_idx].to(torch.float32)
            k_chunk = k[b_idx, rt, h_idx, :].to(torch.float32)
            v_chunk = v[b_idx, rt, h_idx, :].to(torch.float32)
            A_chunk = A[b_idx, rt, h_idx, :].to(torch.float32)

            k_scaled = k_chunk * (beta_vals * torch.exp(g_vals))[:, None]
            v_scaled = v_chunk * beta_vals[:, None]

            w_result = hl.dot(A_chunk, k_scaled, out_dtype=torch.float32)
            u_result = hl.dot(A_chunk, v_scaled, out_dtype=torch.float32)

            w_out[b_idx, rt, h_idx, :] = w_result.to(k.dtype)
            u_out[b_idx, rt, h_idx, :] = u_result.to(v.dtype)

        return w_out, u_out

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

    return k.contiguous(), v.contiguous(), beta.contiguous(), A.contiguous(), g_cumsum.contiguous()


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
    pid_type: str,
    num_warps: int,
    num_stages: int,
    l2_groupings,
    loop_orders,
    load_eviction_policies,
    maxnreg: int | None = None,
    num_sm_multiplier: int | None = None,
    range_multi_buffers=None,
    range_num_stages=None,
    range_unroll_factors=None,
) -> helion.Config:
    kwargs = dict(
        block_sizes=[],
        indexing=indexing,
        pid_type=pid_type,
        num_warps=num_warps,
        num_stages=num_stages,
        l2_groupings=l2_groupings,
        loop_orders=loop_orders,
        load_eviction_policies=load_eviction_policies,
        range_flattens=[None],
        range_multi_buffers=[False] if range_multi_buffers is None else range_multi_buffers,
        range_num_stages=[3] if range_num_stages is None else range_num_stages,
        range_unroll_factors=[4] if range_unroll_factors is None else range_unroll_factors,
    )
    if maxnreg is not None:
        kwargs["maxnreg"] = maxnreg
    if num_sm_multiplier is not None:
        kwargs["num_sm_multiplier"] = num_sm_multiplier
    return helion.Config(**kwargs)


def iter_configs():
    seen: set[str] = set()
    base = dict(
        indexing=["tensor_descriptor", "pointer", "tensor_descriptor", "tensor_descriptor", "tensor_descriptor", "tensor_descriptor", "pointer"],
        pid_type="persistent_blocked",
        num_warps=32,
        num_stages=1,
        l2_groupings=[16],
        loop_orders=[[0, 1]],
        load_eviction_policies=["", "first", "", "first", ""],
        maxnreg=32,
        num_sm_multiplier=16,
        range_multi_buffers=[False],
        range_num_stages=[3],
        range_unroll_factors=[4],
    )

    variants = {
        "baseline": {},
        "nw4": {"num_warps": 4},
        "nw8": {"num_warps": 8},
        "nw16": {"num_warps": 16},
        "l2g1": {"l2_groupings": [1]},
        "l2g4": {"l2_groupings": [4]},
        "l2g8": {"l2_groupings": [8]},
        "nsm1": {"num_sm_multiplier": 1},
        "nsm2": {"num_sm_multiplier": 2},
        "nsm4": {"num_sm_multiplier": 4},
        "nsm8": {"num_sm_multiplier": 8},
        "rns0": {"range_num_stages": [0]},
        "rns1": {"range_num_stages": [1]},
        "ruf0": {"range_unroll_factors": [0]},
        "ruf2": {"range_unroll_factors": [2]},
        "pointer_heavy": {"indexing": ["pointer", "pointer", "pointer", "tensor_descriptor", "pointer", "pointer", "pointer"]},
        "all_pointer": {"indexing": ["pointer"] * 7},
        "flat_nw8": {"pid_type": "flat", "num_warps": 8, "num_sm_multiplier": None, "maxnreg": None, "l2_groupings": [2], "range_multi_buffers": [None], "range_num_stages": [0], "range_unroll_factors": [0], "load_eviction_policies": ["", "", "", "", ""]},
        "flat_nw4": {"pid_type": "flat", "num_warps": 4, "num_sm_multiplier": None, "maxnreg": None, "l2_groupings": [2], "range_multi_buffers": [None], "range_num_stages": [0], "range_unroll_factors": [0], "load_eviction_policies": ["", "", "", "", ""]},
        "pint_nw8_nsm2": {"pid_type": "persistent_interleaved", "num_warps": 8, "num_sm_multiplier": 2, "maxnreg": 64, "l2_groupings": [1], "indexing": ["pointer", "pointer", "pointer", "tensor_descriptor", "pointer", "pointer", "pointer"], "range_multi_buffers": [None], "range_num_stages": [0], "range_unroll_factors": [0], "load_eviction_policies": ["", "", "", "", ""]},
        "pint_nw8_nsm4": {"pid_type": "persistent_interleaved", "num_warps": 8, "num_sm_multiplier": 4, "maxnreg": 64, "l2_groupings": [1], "indexing": ["pointer", "pointer", "pointer", "tensor_descriptor", "pointer", "pointer", "pointer"], "range_multi_buffers": [None], "range_num_stages": [0], "range_unroll_factors": [0], "load_eviction_policies": ["", "", "", "", ""]},
    }

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
