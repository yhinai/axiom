#!/usr/bin/env python3
"""
Targeted H200-safe sweep for causal_conv1d.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import helion
import helion.language as hl
import torch


BENCHMARK_SHAPES = [
    (1, 1536, 2048, 4, 2146),
    (1, 2560, 2048, 4, 3129),
    (1, 2560, 4096, 4, 54352),
]


@dataclass(frozen=True)
class ShapeResult:
    shape: tuple[int, int, int, int]
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


def make_kernel(config: helion.Config, use_clamp: bool):
    @helion.kernel(static_shapes=True, config=config)
    def kernel(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        D = x.size(1)
        S = x.size(2)
        W = hl.specialize(w.size(1))

        y = torch.empty(B, D, S, dtype=x.dtype, device=x.device)

        for rb, rd, rs in hl.tile([B, D, S], block_size=[1, None, None]):
            bi = rb.begin
            acc = hl.zeros([rd, rs], dtype=torch.float32)
            for j in range(W):
                idx = rs.index + j - (W - 1)
                if use_clamp:
                    safe_idx = idx.clamp(min=0)
                    x_val = hl.load(x, [bi, rd, safe_idx]).to(torch.float32)
                    valid = (idx >= 0).to(torch.float32)
                    coeff = w[rd, j].to(torch.float32)
                    acc = acc + x_val * coeff[:, None] * valid[None, :]
                else:
                    x_val = hl.load(x, [bi, rd, idx], extra_mask=idx >= 0).to(torch.float32)
                    acc = acc + x_val * w[rd, j].to(torch.float32)[:, None]
            acc = acc + b[rd].to(torch.float32)[:, None]
            y[rb, rd, rs] = acc[None, :, :].to(y.dtype)

        return y

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


def make_inputs(B: int, D: int, S: int, W: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.randn(B, D, S, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    weight = torch.randn(D, W, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    bias = torch.randn(D, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    return x, weight, bias


def test_config(label: str, config: helion.Config, use_clamp: bool) -> SweepResult:
    try:
        shape_results: list[ShapeResult] = []
        for B, D, S, W, seed in BENCHMARK_SHAPES:
            kernel_fn = make_kernel(config, use_clamp=use_clamp)
            args = make_inputs(B, D, S, W, seed)
            kernel_fn(*args)
            torch.cuda.synchronize()
            median_s = benchmark_kernel(kernel_fn, args)
            shape_results.append(ShapeResult((B, D, S, W), median_s))
        return SweepResult(label=label, config=config, shape_results=tuple(shape_results))
    except Exception as exc:
        return SweepResult(label=label, config=config, shape_results=None, error=str(exc)[:180])


def make_config(*, block_sizes, num_warps, num_stages, l2_groupings=None, loop_orders=None):
    kwargs = dict(block_sizes=block_sizes, num_warps=num_warps, num_stages=num_stages)
    if l2_groupings is not None:
        kwargs["l2_groupings"] = l2_groupings
    if loop_orders is not None:
        kwargs["loop_orders"] = loop_orders
    return helion.Config(**kwargs)


def iter_configs():
    base_variants = [
        ("baseline_mask_4096", make_config(block_sizes=[1, 4096], num_warps=4, num_stages=3, l2_groupings=[8], loop_orders=[[0, 2, 1]]), False),
        ("mask_2048_w4_s3", make_config(block_sizes=[1, 2048], num_warps=4, num_stages=3, l2_groupings=[8], loop_orders=[[0, 2, 1]]), False),
        ("mask_1024_w4_s3", make_config(block_sizes=[1, 1024], num_warps=4, num_stages=3, l2_groupings=[8], loop_orders=[[0, 2, 1]]), False),
        ("mask_512_w1_s1", make_config(block_sizes=[1, 512], num_warps=1, num_stages=1), False),
        ("mask_512_w2_s1", make_config(block_sizes=[1, 512], num_warps=2, num_stages=1), False),
        ("mask_512_w4_s1", make_config(block_sizes=[1, 512], num_warps=4, num_stages=1), False),
        ("mask_512_w1_s2", make_config(block_sizes=[1, 512], num_warps=1, num_stages=2), False),
        ("mask_1024_w1_s1", make_config(block_sizes=[1, 1024], num_warps=1, num_stages=1), False),
        ("clamp_512_w1_s1", make_config(block_sizes=[1, 512], num_warps=1, num_stages=1), True),
        ("clamp_512_w2_s1", make_config(block_sizes=[1, 512], num_warps=2, num_stages=1), True),
        ("clamp_512_w4_s1", make_config(block_sizes=[1, 512], num_warps=4, num_stages=1), True),
        ("clamp_1024_w1_s1", make_config(block_sizes=[1, 1024], num_warps=1, num_stages=1), True),
        ("clamp_2048_w1_s1", make_config(block_sizes=[1, 2048], num_warps=1, num_stages=1), True),
        ("clamp_4096_w1_s1", make_config(block_sizes=[1, 4096], num_warps=1, num_stages=1), True),
        ("clamp_2048_w2_s1", make_config(block_sizes=[1, 2048], num_warps=2, num_stages=1), True),
    ]
    yield from base_variants


def main() -> int:
    print("Shape set:", [shape[:4] for shape in BENCHMARK_SHAPES], flush=True)
    print(f"Device: {torch.cuda.get_device_name()}", flush=True)
    print("=" * 80, flush=True)

    results = [test_config(label, cfg, use_clamp) for label, cfg, use_clamp in iter_configs()]
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
