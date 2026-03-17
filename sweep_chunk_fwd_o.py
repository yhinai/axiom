#!/usr/bin/env python3
"""
Targeted config sweep for gated_deltanet_chunk_fwd_o.

Uses the current fused-accumulator math and searches a curated H200-safe
config neighborhood inspired by the strongest external submissions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import helion
import helion.language as hl
import torch


_LOG2E = 1.4426950408889634


@dataclass(frozen=True)
class SweepResult:
    label: str
    median_s: float | None
    config: helion.Config
    dot_precision: str
    error: str | None = None


def make_kernel(config: helion.Config, dot_precision: str = "tf32"):
    @helion.kernel(static_shapes=True, dot_precision=dot_precision, config=config)
    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
        g: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        B, T, H, K = q.shape
        V = v.shape[-1]
        C = 64
        K = hl.specialize(K)
        V = hl.specialize(V)

        out = torch.empty_like(v)
        BH = B * H

        for flat_bh, tile_t in hl.tile([BH, T], block_size=[1, C]):
            b_idx = flat_bh.begin // H
            h_idx = flat_bh.begin % H
            c_idx = tile_t.begin // C

            g_vals = g[b_idx, tile_t, h_idx].to(torch.float32)
            q_chunk = q[b_idx, tile_t, h_idx, :].to(torch.float32)
            k_chunk = k[b_idx, tile_t, h_idx, :].to(torch.float32)
            v_chunk = v[b_idx, tile_t, h_idx, :]

            qk = hl.dot(q_chunk, k_chunk.T)
            g_diff = g_vals[:, None] - g_vals[None, :]
            qk = qk * torch.exp2(g_diff * _LOG2E)
            idx = hl.arange(tile_t.block_size)
            mask = idx[:, None] >= idx[None, :]
            qk = torch.where(mask, qk, 0.0)

            acc = hl.dot(qk.to(v.dtype), v_chunk)
            q_g = q_chunk * torch.exp2(g_vals[:, None] * _LOG2E)
            acc = hl.dot(q_g, h[b_idx, c_idx, h_idx, :, :], acc=acc)

            out[b_idx, tile_t, h_idx, :] = (acc * scale).to(out.dtype)

        return out

    return kernel


def benchmark_kernel(kernel_fn, args, warmup: int = 10, iters: int = 30) -> float:
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


def test_config(
    label: str,
    config: helion.Config,
    dot_precision: str,
    args: tuple[torch.Tensor, ...],
) -> SweepResult:
    try:
        kernel_fn = make_kernel(config, dot_precision=dot_precision)
        kernel_fn(*args)
        torch.cuda.synchronize()
        median_s = benchmark_kernel(kernel_fn, args)
        return SweepResult(label=label, median_s=median_s, config=config, dot_precision=dot_precision)
    except Exception as exc:  # pragma: no cover - sweep helper
        return SweepResult(
            label=label,
            median_s=None,
            config=config,
            dot_precision=dot_precision,
            error=str(exc)[:160],
        )


def make_config(
    *,
    indexing=None,
    pid_type: str = "flat",
    num_warps: int = 16,
    num_stages: int = 1,
    l2_groupings=None,
    loop_orders=None,
    load_eviction_policies=None,
    range_flattens=None,
    range_multi_buffers=None,
    range_num_stages=None,
    range_unroll_factors=None,
    maxnreg: int | None = None,
    num_sm_multiplier: int | None = None,
) -> helion.Config:
    if indexing is None:
        indexing = ["tensor_descriptor", "pointer", "pointer", "pointer", "pointer", "pointer"]
    if l2_groupings is None:
        l2_groupings = [1]
    if loop_orders is None:
        loop_orders = [[1, 0]]
    if load_eviction_policies is None:
        load_eviction_policies = ["", "", "", "", ""]
    if range_flattens is None:
        range_flattens = [None]
    if range_multi_buffers is None:
        range_multi_buffers = [None]
    if range_num_stages is None:
        range_num_stages = [0]
    if range_unroll_factors is None:
        range_unroll_factors = [0]

    kwargs = dict(
        block_sizes=[],
        indexing=indexing,
        pid_type=pid_type,
        num_warps=num_warps,
        num_stages=num_stages,
        l2_groupings=l2_groupings,
        loop_orders=loop_orders,
        load_eviction_policies=load_eviction_policies,
        range_flattens=range_flattens,
        range_multi_buffers=range_multi_buffers,
        range_num_stages=range_num_stages,
        range_unroll_factors=range_unroll_factors,
    )
    if maxnreg is not None:
        kwargs["maxnreg"] = maxnreg
    if num_sm_multiplier is not None:
        kwargs["num_sm_multiplier"] = num_sm_multiplier
    return helion.Config(**kwargs)


def iter_configs():
    seen: set[str] = set()

    def add(label: str, config: helion.Config, dot_precision: str = "tf32"):
        key = f"{dot_precision}|{repr(config)}"
        if key not in seen:
            seen.add(key)
            yield label, config, dot_precision

    base = make_config()
    yield from add("baseline", base)

    for dp in ("tf32", "ieee"):
        yield from add(f"dot_precision={dp}", base, dp)

    for nw in (4, 8, 16, 32):
        yield from add(f"num_warps={nw}", make_config(num_warps=nw))

    for ns in (1, 2, 3, 5):
        yield from add(f"num_stages={ns}", make_config(num_stages=ns))

    for lg in (1, 2, 4, 8):
        yield from add(f"l2_groupings={lg}", make_config(l2_groupings=[lg]))

    indexing_variants = {
        "current_mixed": ["tensor_descriptor", "pointer", "pointer", "pointer", "pointer", "pointer"],
        "all_pointer": ["pointer"] * 6,
        "pointer_qkv_h": ["pointer", "pointer", "pointer", "pointer", "pointer", "pointer"],
        "tensor_heavy": ["tensor_descriptor", "tensor_descriptor", "pointer", "pointer", "pointer", "pointer"],
        "desu_like": ["tensor_descriptor", "pointer", "pointer", "pointer", "pointer", "pointer"],
    }
    for name, indexing in indexing_variants.items():
        yield from add(f"indexing={name}", make_config(indexing=indexing))

    for pid_type in ("persistent_blocked", "persistent_interleaved"):
        for nw in (8, 16):
            for nsm in (1, 2, 4, 8):
                for maxnreg in (32, 64):
                    yield from add(
                        f"{pid_type}_nw{nw}_nsm{nsm}_r{maxnreg}",
                        make_config(
                            pid_type=pid_type,
                            num_warps=nw,
                            num_stages=1,
                            num_sm_multiplier=nsm,
                            maxnreg=maxnreg,
                            indexing=["pointer"] * 6,
                        ),
                    )


def main() -> int:
    torch.manual_seed(2146)
    device = "cuda"

    B, T, H, K, V = 2, 1024, 3, 64, 64
    q = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    k = torch.randn(B, T, H, K, device=device, dtype=torch.float32)
    v = torch.randn(B, T, H, V, device=device, dtype=torch.float32)
    h = torch.randn(B, T // 64, H, K, V, device=device, dtype=torch.float32)
    g = torch.randn(B, T, H, device=device, dtype=torch.float32)
    scale = K ** -0.5
    args = (q, k, v, h, g, scale)

    print(f"Shape: B={B}, T={T}, H={H}, K={K}, V={V}", flush=True)
    print(f"Device: {torch.cuda.get_device_name()}", flush=True)
    print("=" * 80, flush=True)

    results: list[SweepResult] = []
    for idx, (label, config, dot_precision) in enumerate(iter_configs(), start=1):
        result = test_config(label, config, dot_precision, args)
        results.append(result)
        if result.median_s is None:
            print(f"[{idx:03d}] FAIL {label}: {result.error}", flush=True)
        else:
            print(f"[{idx:03d}] {label}: {result.median_s * 1e6:.1f} us", flush=True)

    good = [result for result in results if result.median_s is not None]
    good.sort(key=lambda item: item.median_s)

    print("\n=== Top Results ===", flush=True)
    for rank, result in enumerate(good[:10], start=1):
        print(
            f"{rank:02d}. {result.label}: {result.median_s * 1e6:.1f} us "
            f"[{result.dot_precision}] {result.config!r}",
            flush=True,
        )

    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
