# Triton-to-tile-IR: Deep Analysis & Integration Guide

## For the Helion GPU Kernel Hackathon on NVIDIA B200 (Blackwell)

---

# Table of Contents

1. [What is TileIR?](#1-what-is-tileir)
2. [Why TileIR Matters for Performance](#2-why-tileir-matters-for-performance)
3. [Compilation Pipeline: TileIR vs Standard Triton](#3-compilation-pipeline-tileir-vs-standard-triton)
4. [Installation](#4-installation)
5. [Activation & Configuration](#5-activation--configuration)
6. [New Config Parameters](#6-new-config-parameters)
7. [Modified & Removed Parameters](#7-modified--removed-parameters)
8. [IR Operations & Conversion Mappings](#8-ir-operations--conversion-mappings)
9. [The tileiras Compiler Internals](#9-the-tileiras-compiler-internals)
10. [Performance Benchmarks](#10-performance-benchmarks)
11. [Performance Tuning Recipes](#11-performance-tuning-recipes)
12. [Helion Integration Details](#12-helion-integration-details)
13. [Autotuning with TileIR](#13-autotuning-with-tileir)
14. [Limitations & Unsupported Features](#14-limitations--unsupported-features)
15. [Hackathon Strategy: Applying TileIR to Competition Kernels](#15-hackathon-strategy-applying-tileir-to-competition-kernels)
16. [Environment Variables Reference](#16-environment-variables-reference)
17. [Relationship to CUDA Ecosystem](#17-relationship-to-cuda-ecosystem)
18. [Sources](#18-sources)

---

# 1. What is TileIR?

## 1.1 Core Concept

**CUDA Tile IR** is an MLIR-based intermediate representation introduced in **CUDA 13.1** that models the GPU as a **tile-based processor** rather than using the traditional SIMT (Single Instruction, Multiple Threads) model. Instead of lowering tile-level abstractions to thread-level code (as standard Triton does), Tile IR **preserves tile-level semantics throughout the entire compilation pipeline**, letting the hardware-aware compiler (`tileiras`) handle all thread mapping, register allocation, and instruction scheduling automatically.

**Triton-to-tile-IR** (https://github.com/triton-lang/Triton-to-tile-IR) is an incubator repository by the triton-lang organization that adds CUDA Tile IR as a backend for the OpenAI Triton compiler. It serves as a bridge: users write kernels in Triton (or Helion, which compiles to Triton), and instead of the standard compilation path, the backend routes through Tile IR to produce optimized GPU binaries.

## 1.2 Repository Details

| Attribute | Value |
|-----------|-------|
| **URL** | https://github.com/triton-lang/Triton-to-tile-IR |
| **Organization** | triton-lang |
| **Created** | December 13, 2025 |
| **License** | MIT |
| **Primary Language** | MLIR |
| **Stars** | 117 |
| **Size** | ~85 MB |
| **Latest Release** | v3.6.0-rc1 (March 11, 2026) |

## 1.3 The Name

The naming parallels the Helion/Triton convention from nuclear physics:
- **Triton** = hydrogen-3 nucleus (the existing Triton compiler)
- **Helion** = helium-3 nucleus (the high-level DSL)
- **Tile IR** = the intermediate representation that preserves tile-level abstractions

## 1.4 Key Insight

Traditional GPU compilation decomposes tile operations into individual thread instructions early in the pipeline, losing optimization opportunities. Tile IR defers this decomposition to the final compiler stage (`tileiras`), which has full knowledge of the target hardware (Blackwell tensor core layout, memory hierarchy, warp scheduling). This enables optimizations that are impossible or impractical in the standard pipeline.

---

# 2. Why TileIR Matters for Performance

## 2.1 Performance Gains

On NVIDIA B200 (Blackwell, 1000W):

| Benchmark | PTX Backend | TileIR Backend | Improvement |
|-----------|-------------|----------------|-------------|
| Persistent Matmul (K=8192) | 517 TFLOPS | 648 TFLOPS | **+25%** |
| Persistent Matmul (K=4096) | 524 TFLOPS | 633 TFLOPS | **+21%** |
| Flash Attention (1K seq) | ~340 TFLOPS | 548 TFLOPS | **+61%** |
| Flash Attention (16K seq) | ~570 TFLOPS | 918 TFLOPS | **+61%** |
| FlashAttention-4 peak | -- | 1,605 TFLOPS | **71% of theoretical max** |

## 2.2 Why It's Faster

1. **Tile-level optimization**: tileiras can optimize at the tile level (tensor core scheduling, shared memory banking, register allocation for tiles) rather than thread-level
2. **2CTA MMA mode**: Blackwell supports a special 2-CTA cooperative MMA mode that doubles tensor core throughput for large matmuls -- TileIR's `num_ctas=2` enables this directly
3. **Automatic warp scheduling**: tileiras determines optimal warp configuration internally (vs manual `num_warps` tuning in PTX backend)
4. **TMA integration**: Native support for Tensor Memory Accelerator (TMA) hardware for high-bandwidth memory transfers
5. **30+ internal optimization passes**: tileiras includes passes for layout assignment, schedule generation, async materialization, CTA planning, allocation optimization, loop unrolling, slice-and-fuse, dynamic persistent kernels, and more

---

# 3. Compilation Pipeline: TileIR vs Standard Triton

## 3.1 Standard Triton (NVIDIA PTX Backend)

```
Triton Python Source
    |
    v
[TTIR] Triton IR (MLIR)
    |  Passes: inliner, combine, canonicalizer, CSE, LICM, dead code
    v
[TTGIR] TritonGPU IR (MLIR)
    |  Passes: warp_specialize, pipeline, allocate_shared_memory,
    |          allocate_tensor_memory, nvgpu_to_llvm, nvvm_to_llvm
    |  Thread-level decomposition happens here
    v
[LLIR] LLVM IR
    |
    v
[PTX] PTX Assembly
    |  via ptxas
    v
[CUBIN] GPU Binary
```

**5 stages.** Thread mapping is explicit in TTGIR. Many optimization decisions (warp count, pipeline depth, shared memory layout) are fixed at the TTGIR level.

## 3.2 TileIR Backend

```
Triton Python Source
    |
    v
[TTIR] Triton IR (MLIR)
    |  Passes: inliner, combine, canonicalizer, CSE, LICM, dead code
    v
[CUDA Tile IR] Tile-level IR (MLIR)
    |  Passes:
    |    1. lift_tt_cf_to_scf (control flow to SCF)
    |    2. assume_to_tileir (assume ops conversion)
    |    3. triton_to_cudatile (MAIN CONVERSION PASS)
    |       Parameters: approx, FTZ, capability, num_ctas, occupancy, num_stages
    |    4. auto_gen_memtoken (memory aliasing tokens)
    |    5. fma_fusion (fused multiply-add)
    |    6. strip_debuginfo
    |    7. dialect verification (only cuda_tile ops remain)
    |    8. bytecode serialization (.tilebc)
    v
[CUBIN] GPU Binary
    via tileiras --gpu-name=sm_100 --opt-level=3
```

**3 stages.** Tile-level semantics are preserved until tileiras. Thread mapping, register allocation, and instruction scheduling are all handled by tileiras internally through 30+ optimization passes.

## 3.3 Comparison Table

| Aspect | Standard Triton (PTX) | TileIR Backend |
|--------|----------------------|----------------|
| **Pipeline stages** | 5 (TTIR → TTGIR → LLIR → PTX → CUBIN) | 3 (TTIR → TileIR → CUBIN) |
| **Thread mapping** | Explicit in TritonGPU IR | Deferred to tileiras compiler |
| **Warp scheduling** | `num_warps` is tunable | Ignored; compiler decides automatically |
| **Pipeline stages** | Strict directive | Cost hint; compiler determines optimal |
| **Warp specialization** | Supported | Not available |
| **Assembler** | `ptxas` | `tileiras` (embeds ptxas internally) |
| **Memory model** | Ordered by default | Unordered by default (global memory) |
| **FP precision** | Approx/FTZ on by default | Disabled by default, opt-in via env vars |
| **Cached output** | `.cubin` | `.tileIR` |
| **Fallback** | N/A | Automatic fallback to PTX on failure |

---

# 4. Installation

## 4.1 Option A: Pre-built Wheel (Recommended)

The release `v3.6.0-rc1` (published March 11, 2026) provides pre-built wheels:

```bash
# Python 3.12
pip install https://github.com/triton-lang/Triton-to-tile-IR/releases/download/v3.6.0-rc1/nvtriton-3.6.0-cp312-cp312-linux_x86_64.whl

# Python 3.13
pip install https://github.com/triton-lang/Triton-to-tile-IR/releases/download/v3.6.0-rc1/nvtriton-3.6.0-cp313-cp313-linux_x86_64.whl
```

The wheels (~785 MB) embed `tileiras`, `ptxas`, and `libnvvm` binaries -- no external CUDA_HOME required.

## 4.2 Option B: From Source

```bash
git clone https://github.com/triton-lang/Triton-to-tile-IR.git
cd Triton-to-tile-IR
pip install -e .
```

## 4.3 Option C: Side-by-Side Install

Install nvtriton to an isolated directory alongside upstream Triton (avoids conflicts):

```bash
NVTRITON_DIR=$VIRTUAL_ENV/opt/nvtriton
mkdir -p $NVTRITON_DIR
pip install --no-cache-dir --no-deps --target $NVTRITON_DIR \
    ./nvtriton-3.6.0-cp312-cp312-linux_x86_64.whl

# Activate by prepending to PYTHONPATH
export PYTHONPATH=$NVTRITON_DIR
export ENABLE_TILE=1
python my_script.py
```

This works because `PYTHONPATH` is searched before `site-packages`, shadowing the upstream triton package.

## 4.4 Requirements

| Requirement | Version |
|-------------|---------|
| **GPU** | NVIDIA Blackwell (sm_100, sm_120) |
| **CUDA** | 13.1+ (embedded in wheel) |
| **Driver** | r580+ |
| **Python** | 3.10 - 3.13 |
| **PyTorch** | 2.9+ |
| **OS** | Linux x86_64 only |

---

# 5. Activation & Configuration

## 5.1 Environment Variables (Must Be Set Before Imports)

```python
import os
os.environ["ENABLE_TILE"] = "1"          # Required: enables TileIR driver
os.environ["HELION_BACKEND"] = "tileir"  # Required: tells Helion to use TileIR

# Optional performance trade-offs:
os.environ["TILEIR_ENABLE_APPROX"] = "1"  # Approximate math (fast exp, etc.)
os.environ["TILEIR_ENABLE_FTZ"] = "1"     # Flush-to-zero for denormals

import helion
import helion.language as hl
```

## 5.2 Verification

```python
from helion._compat import use_tileir_tunables
assert use_tileir_tunables(), "TileIR not active -- check ENABLE_TILE and GPU capability"
```

## 5.3 How Helion Detects TileIR

From `helion/_compat.py`, `use_tileir_tunables()` checks:
1. CUDA must be available
2. GPU compute capability must be **10.x or 12.x** (Blackwell)
3. Triton GPU target's `backend` attribute must equal `"tileir"`

If any check fails, TileIR features are silently disabled and standard Triton backend is used.

## 5.4 Decorator-Level Configuration

```python
@helion.kernel(
    backend="tileir",
    config=helion.Config(
        block_sizes=[128, 128],
        num_ctas=2,
        occupancy=2,
        num_stages=4,
        indexing="tensor_descriptor",
    ),
)
def my_kernel(x: torch.Tensor) -> torch.Tensor:
    ...
```

---

# 6. New Config Parameters

TileIR introduces two new parameters to `helion.Config`:

## 6.1 `num_ctas` (Cooperative Thread Array Count)

| Attribute | Value |
|-----------|-------|
| **Type** | Power of 2 |
| **Default** | 1 |
| **Tuning Range** | 1-2 (up to 16 configurable) |
| **Autotuner** | PowerOfTwoFragment(1, 2) |

**What it does**: Controls the number of CTAs (thread blocks) in one CGA (Cooperative Grid Array). On Blackwell, setting `num_ctas=2` enables **2-CTA cooperative MMA mode**, which doubles tensor core throughput for dense matrix operations.

**When to use `num_ctas=2`**:
- Any kernel with `hl.dot` operations (matmul, attention QK^T, attention PV)
- Gated DeltaNet kernels (chunk_fwd_h, chunk_fwd_o, recompute_w_u) -- all use `hl.dot`
- NOT beneficial for purely elementwise kernels (FP8 quantization)

**Critical insight**: `num_ctas=2` is the single most impactful TileIR parameter for dot-heavy kernels.

## 6.2 `occupancy` (SM Occupancy Hint)

| Attribute | Value |
|-----------|-------|
| **Type** | Power of 2 |
| **Default** | 1 |
| **Tuning Range** | 1-8 (up to 32 configurable) |
| **Autotuner** | PowerOfTwoFragment(1, 8) |

**What it does**: Controls expected simultaneous thread blocks per SM. Higher values can hide memory latency by having more warps ready to execute, but at the cost of reduced per-block resources (registers, shared memory).

**Tuning guidance**:
- **occupancy=1**: Maximum resources per block. Best for compute-heavy kernels with large register usage.
- **occupancy=2-4**: Good balance for mixed compute/memory kernels. Recommended starting point.
- **occupancy=4-8**: Best for memory-bound kernels that benefit from latency hiding.

---

# 7. Modified & Removed Parameters

## 7.1 Modified Parameters Under TileIR

| Parameter | Standard Backend | TileIR Backend | Reason |
|-----------|-----------------|---------------|--------|
| `num_warps` | Tunable (power of 2: 1-32) | **Fixed at 4** (not tunable) | tileiras determines optimal warps internally. CUDA 13.1 Tile IR doesn't expose warp control. |
| `num_stages` | IntegerFragment (1-8) | **EnumFragment (1-10)** | Treated as a cost hint, not strict directive. Discrete search is more effective for TileIR. |
| `indexing` | "pointer", "block_ptr", "tensor_descriptor" | "pointer", "tensor_descriptor" only | `block_ptr` is **not supported** by TileIR. |

## 7.2 Removed/Unsupported Parameters

These are excluded from TileIR autotuning search space entirely:

| Parameter | Reason |
|-----------|--------|
| `static_ranges` | TileIR handles loop optimization internally |
| `range_unroll_factors` | tileiras manages unrolling |
| `range_multi_buffers` | tileiras manages buffering |
| `range_flattens` | tileiras manages loop structure |
| `range_warp_specialize` | Not supported on TileIR |
| `load_eviction_policies` | TileIR memory model handles caching internally |

**When porting configs from standard Triton to TileIR**: Remove all unsupported parameters to avoid errors. The autotuner will automatically exclude them.

---

# 8. IR Operations & Conversion Mappings

The main conversion pass (`TritonToTileIRPass.cpp`, 105KB) implements these key mappings:

## 8.1 Arithmetic Operations

| Triton (arith dialect) | CUDA Tile IR |
|----------------------|--------------|
| `AddFOp`, `SubFOp`, `MulFOp`, `DivFOp` | Corresponding cuda_tile float ops |
| `AddIOp`, `SubIOp`, `MulIOp` | cuda_tile int ops (auto-upcast i8 to i16) |
| `CmpIOp`, `CmpFOp` | Predicate comparison ops |
| `AndIOp`, `OrIOp`, `XOrIOp`, `ShLIOp`, `ShRIOp` | Bitwise ops |
| `ExtSIOp`, `ExtUIOp`, `TruncIOp` | Type conversion ops |

## 8.2 Math/Transcendental

All mapped through `ConvertExternElementwiseOp`:
`exp`, `exp2`, `log2`, `sin`, `cos`, `tan`, `tanh`, `sqrt`, `rsqrt`, `ceil`, `floor`, `pow`, `abs`

## 8.3 Memory Operations

| Triton | CUDA Tile IR | Notes |
|--------|-------------|-------|
| `LoadOp` | `LoadPtrTkoOp` | With token generation and memory semantics |
| `StoreOp` | `StorePtrTkoOp` | With token and ordering |
| TMA descriptor loads | `MakeTensorViewOp` + `MakePartitionViewOp` | Index divides by tile size |
| `MakeTensorDescriptor` | View chain with 16-byte alignment | Wraps view operations |

## 8.4 Dot Product (Matrix Multiply)

| Mode | Conversion |
|------|-----------|
| IEEE (default) | Direct `MmaFOp`/`MmaIOp` lowering |
| TF32 | Splits F32 inputs into lower-precision MMAs |
| BF16x3, BF16x6 | Multiple accumulated lower-precision MMAs |

NaN replacement is added for reduced-precision modes.

## 8.5 Reductions and Scans

Identity values are extracted from combine blocks. Recognized identities: add (ZERO), mul (ONE), bitwise ops, min/max with proper signed/unsigned/float handling.

## 8.6 Control Flow

| Triton | CUDA Tile IR |
|--------|-------------|
| `scf::ForOp` | `cuda_tile::ForOp` with yield → `ContinueOp` |
| `scf::IfOp` | `cuda_tile::IfOp` with region inlining |
| `scf::WhileOp` | `cuda_tile::LoopOp` + `IfOp` + `ContinueOp`/`BreakOp` |

---

# 9. The tileiras Compiler Internals

## 9.1 Internal Pipeline (6 stages)

```
[1] cuda_tile dialect
    Architecture-independent tensor ops with abstract views
        |
        v
[2] nv_tileaa dialect
    Tile-level ops with explicit memory references
        |
        v
[3] nv_tileas dialect
    Scheduled ops with async pipelines and tensor core instructions
        |
        v
[4] LLVM/NVVM IR
    Standard LLVM IR with NVIDIA intrinsics
        |
        v
[5] PTX (internal)
    Generated by LLVM/NVVM backend
        |
        v
[6] SASS
    Native GPU machine code (via internal ptxas)
```

## 9.2 Optimization Passes (30+)

tileiras includes the following documented optimization passes:

1. **Layout assignment** for dot, load, store operations
2. **Schedule generation** for optimal instruction ordering
3. **Async materialization** for overlapping compute and memory
4. **CTA planning** for cooperative multi-CTA execution
5. **Allocation optimization** for register and shared memory
6. **Loop unrolling** with configurable thresholds
7. **Slice-and-fuse** for operation fusion
8. **Dynamic persistent kernels** for improved GPU utilization
9. **OCG knob insertion** for runtime tuning
10. **FMA fusion** for multiply-add operations
11. **Memory token generation** for aliasing safety
12. **TMA coordination** for tensor memory accelerator
13. **Register pressure management**
14. **Shared memory banking optimization**
15. **Warp scheduling optimization**

---

# 10. Performance Benchmarks

## 10.1 Persistent Matmul on B200 (1000W)

Using `matmul_kernel_descriptor_persistent` (TMA-based):

| K Dimension | PTX Backend (TFLOPS) | TileIR Backend (TFLOPS) | Speedup |
|-------------|---------------------|------------------------|---------|
| 4096 | 524.3 | 632.6 | **+21%** |
| 8192 | 517.0 | 648.0 | **+25%** |

## 10.2 Flash Attention on B200

| Sequence Length | TFLOPS | Notes |
|----------------|--------|-------|
| 1K | 548 | With full autotuning |
| 16K | 918 | With full autotuning |
| Peak (FlashAttention-4) | 1,605 | 71% of theoretical max |

## 10.3 Key Performance Insights

1. **TMA is essential**: Using `tl.load` (pointer-based) is **20%+ slower** than TMA APIs (`tensor_descriptor` in Helion). Always prefer `tensor_descriptor` indexing for matmul/attention kernels.

2. **2CTA mode is critical**: `num_ctas=2` enables cooperative MMA that can double tensor core utilization for dense dot workloads.

3. **Fast math matters**: Enabling `TILEIR_ENABLE_APPROX=1` and `TILEIR_ENABLE_FTZ=1` can yield **34-72% speedups** for attention/softmax kernels, with acceptable precision tradeoffs (disabled by default in TileIR, unlike PTX).

4. **The "trap-and-rescue" pattern**: Larger tiles (e.g., 256x128) initially degrade performance by 18-43% due to register pressure, reducing occupancy from 25% to 18.75%. Fast math optimizations "rescue" this by making per-instruction compute efficient enough to compensate.

5. **Standard pointer-based matmul** is actually ~7% slower on TileIR vs PTX (540 vs 503 TFLOPS at K=4096). The gains come specifically from TMA-based kernels.

---

# 11. Performance Tuning Recipes

## 11.1 Per-Kernel-Type Recipes (from NVIDIA's official guide)

### Elementwise Kernels (e.g., FP8 Quantization)
```python
helion.Config(
    block_sizes=[128, 128],
    num_stages=3,
    num_ctas=1,        # No dot operations, CTA cooperation not needed
    occupancy=4,        # Memory-bound, benefit from latency hiding
    indexing="pointer",  # Irregular access patterns OK
)
```

### GEMM / Dense MatMul
```python
helion.Config(
    block_sizes=[128, 128, 32],      # Or [128, 256, 64] for large M,N
    indexing="tensor_descriptor",      # TMA for maximum bandwidth
    num_stages=4,                      # Deep pipeline
    num_ctas=2,                        # 2CTA cooperative MMA mode
    occupancy=2,                       # Balance compute vs resources
)
```

### Large GEMM (M, N >= 4096)
```python
helion.Config(
    block_sizes=[128, 256, 64],
    indexing="tensor_descriptor",
    num_stages=4,
    num_ctas=2,
    occupancy=2,
)
```

### Flash Attention
```python
helion.Config(
    block_sizes=[1, 128, 128],
    indexing="tensor_descriptor",
    num_stages=3,
    num_ctas=1,
    occupancy=2,
)
```

### Softmax / Row Reduction
```python
helion.Config(
    block_sizes=[128, 1024],
    num_stages=2,
    num_ctas=1,
    occupancy=4,        # Memory-bound
    indexing="pointer",  # Row-wise access
)
```

### Layer/RMS Normalization
```python
helion.Config(
    block_sizes=[4, 1024],
    num_stages=2,
    num_ctas=1,
    occupancy=4,
    indexing="pointer",
)
```

## 11.2 General Tuning Heuristics

1. **Start with `num_ctas=2`** for any kernel using `hl.dot`
2. **Start with `occupancy=2`** and search {1, 2, 4, 8}
3. **Use `tensor_descriptor` indexing** for regular, strided access (matmul, attention)
4. **Use `pointer` indexing** for irregular access (elementwise, reductions)
5. **Try `num_stages` 1-10** -- it's a cost hint, so broader search helps
6. **Enable fast math** for attention/softmax: `TILEIR_ENABLE_APPROX=1`, `TILEIR_ENABLE_FTZ=1`
7. **Increase block_sizes** -- TileIR benefits from larger tiles more than PTX backend
8. **Remove unsupported params** when migrating from PTX configs

---

# 12. Helion Integration Details

## 12.1 Detection Logic (`helion/_compat.py`)

```python
def use_tileir_tunables() -> bool:
    """Returns True if TileIR backend is active and GPU supports it."""
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    if capability[0] not in (10, 12):  # Blackwell only
        return False
    # Check if Triton's active target uses tileir backend
    target = get_current_target()
    return getattr(target, 'backend', None) == 'tileir'
```

## 12.2 Settings Enforcement (`helion/runtime/settings.py`)

When `backend == "tileir"`:
- Validates `ENABLE_TILE=1` is set, raises `MissingEnableTile` otherwise
- Adjusts autotuning parameters for TileIR's config space
- Selects TileIR-compatible code generation path

## 12.3 Autotuner ConfigSpec Changes (`helion/autotuner/config_spec.py`)

When `use_tileir_tunables()` returns `True`:

**Added to search space:**
- `num_ctas`: PowerOfTwoFragment(1, 2)
- `occupancy`: PowerOfTwoFragment(1, 8)

**Removed from search space:**
- `static_ranges`, `range_unroll_factors`, `range_multi_buffers`
- `range_flattens`, `range_warp_specialize`, `load_eviction_policies`

**Modified:**
- `num_warps`: Fixed at 4 (not searched)
- `num_stages`: Changed from IntegerFragment to EnumFragment(1-10)
- `indexing`: `block_ptr` removed from options

## 12.4 Fallback Behavior

When TileIR compilation fails (e.g., unsupported operation), it raises `HitFallback` exception and automatically falls back to the NVIDIA PTX backend. This means your kernel will still work, just without TileIR optimizations.

---

# 13. Autotuning with TileIR

## 13.1 Expanded Search Space

With TileIR, the autotuner explores these dimensions:

| Dimension | Values |
|-----------|--------|
| `block_sizes` | Powers of 2 (hardware-aware min/max) |
| `num_stages` | {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} |
| `num_ctas` | {1, 2} |
| `occupancy` | {1, 2, 4, 8} |
| `indexing` | {"pointer", "tensor_descriptor"} |
| `pid_type` | {"flat", "persistent_blocked", "persistent_interleaved"} |
| `l2_groupings` | Powers of 2 |
| `loop_orders` | Permutations |
| `flatten_loops` | {True, False} |

## 13.2 Practical Autotuning Commands

```python
# Full autotuning (recommended for competition)
@helion.kernel(autotune_effort="full")
def my_kernel(x):
    ...

# Quick iteration during development
@helion.kernel(autotune_effort="quick")
def my_kernel(x):
    ...

# Manual config sweep (fastest iteration)
@helion.kernel(configs=[
    helion.Config(block_sizes=[64, 64], num_ctas=1, occupancy=1),
    helion.Config(block_sizes=[64, 64], num_ctas=2, occupancy=2),
    helion.Config(block_sizes=[128, 128], num_ctas=2, occupancy=4),
])
def my_kernel(x):
    ...
```

## 13.3 Recommended Autotuning Timeout

TileIR compilation can be slower than PTX. Recommended:
```bash
export HELION_AUTOTUNE_COMPILE_TIMEOUT=20  # Lower than default 60s
```

This ensures the autotuner doesn't waste time on configs that are slow to compile.

---

# 14. Limitations & Unsupported Features

## 14.1 Hardware Limitations

- **Blackwell only**: sm_100 and sm_120. Does NOT work on Hopper (sm_90), Ada (sm_89), or Ampere (sm_80)
- Future CUDA releases may expand compatibility

## 14.2 Unsupported Triton Operations

| Operation | Status |
|-----------|--------|
| `tt.elementwise_inline_asm` | Not supported |
| `cf.cond_br` (conditional branching) | Not supported |
| `tt.gather`, `tt.unsplat` | Not supported |
| `tt.dot_scaled` | Not supported |
| TMA scatter, gather, reduce | Not supported |
| `math.erf` | Not supported |
| `atomic_rmw` (bf16) | Not supported |
| `atomic_cas` (bf16/fp16) | Not supported |
| i64 index types for memref | Not supported |
| `indexing="block_ptr"` | Not supported |

## 14.3 Helion Feature Limitations

| Feature | Standard Backend | TileIR Backend |
|---------|-----------------|---------------|
| `hl.inline_asm_elementwise` | Supported | **Not supported** |
| `hl.dot_scaled` | Supported | **Not supported** |
| Warp specialization | Supported | **Not available** |
| `block_ptr` indexing | Supported | **Not available** |
| Range config options | Supported | **Not available** |
| `num_warps` tuning | Supported | **Fixed at 4** |

## 14.4 Known Performance Gaps

- Standard pointer-based (`tl.load`) kernels can be ~7% **slower** on TileIR vs PTX
- Small GEMM kernels may underperform
- Large reduction dimensions may cause register spilling (no `num_warps` control)
- Explicit `exp` rewriting may be needed for performance parity in CUDA 13.1

---

# 15. Hackathon Strategy: Applying TileIR to Competition Kernels

## 15.1 Setup (First Thing on B200)

```bash
# Install TileIR backend
pip install https://github.com/triton-lang/Triton-to-tile-IR/releases/download/v3.6.0-rc1/nvtriton-3.6.0-cp312-cp312-linux_x86_64.whl

# Enable TileIR
export ENABLE_TILE=1
export HELION_BACKEND=tileir
export HELION_AUTOTUNE_COMPILE_TIMEOUT=20
```

## 15.2 Per-Kernel TileIR Strategy

### Causal Conv1D (1000 Performance Points)

TileIR benefit: **Low-Medium**. Dominated by small inner loops (W=3-8), not large matmuls.
- Keep `num_ctas=1` (no large dot operations)
- Use `occupancy=4` for memory-bound access pattern
- The filter width loop is small and sequential

```python
# TileIR config for causal_conv1d
helion.Config(
    block_sizes=[1, 256],
    num_ctas=1,
    occupancy=4,
    num_stages=2,
    indexing="pointer",
)
```

### Gated DeltaNet chunk_fwd_h (1000 Performance Points)

TileIR benefit: **High**. Contains `hl.dot` for w@state and k^T@diff.
- Use `num_ctas=2` for 2CTA cooperative MMA
- Use `occupancy=2` for balance between compute and resources
- State matrix [K, V] fits in registers for K,V <= 128

```python
# TileIR config for chunk_fwd_h
helion.Config(
    block_sizes=[],
    num_ctas=2,
    occupancy=2,
    num_stages=4,
    indexing="tensor_descriptor",
)
```

### Gated DeltaNet chunk_fwd_o (1000 Performance Points)

TileIR benefit: **Highest**. Three `hl.dot` operations: QK^T, sim@V, Q@h.
- Use `num_ctas=2` for cooperative MMA
- Use `occupancy=2`
- Enable fast math: `TILEIR_ENABLE_APPROX=1` (exp operations in gating)

```python
# TileIR config for chunk_fwd_o
helion.Config(
    block_sizes=[],
    num_ctas=2,
    occupancy=2,
    num_stages=4,
    indexing="tensor_descriptor",
)
```

### Gated DeltaNet recompute_w_u (1000 Performance Points)

TileIR benefit: **Highest**. Two `hl.dot` operations: A@k_scaled, A@v_scaled.
- Use `num_ctas=2` for cooperative MMA
- C=64 chunk size means [64, 64] matmuls -- good tensor core fit
- Use `occupancy=2-4`

```python
# TileIR config for recompute_w_u
helion.Config(
    block_sizes=[],
    num_ctas=2,
    occupancy=2,
    num_stages=4,
    indexing="tensor_descriptor",
)
```

## 15.3 Dual-Backend Strategy

For maximum competitiveness, **autotune on both backends** and submit the faster config:

```python
import os

# Run 1: TileIR backend
os.environ["ENABLE_TILE"] = "1"
os.environ["HELION_BACKEND"] = "tileir"
# ... autotune, save best TileIR config

# Run 2: Standard Triton backend
del os.environ["ENABLE_TILE"]
os.environ["HELION_BACKEND"] = "triton"
# ... autotune, save best Triton config

# Compare and submit the faster one
```

## 15.4 ACF Files + TileIR

The pre-tuned ACF files at `/opt/booster_pack/` work with the standard Triton backend (`advanced_controls_file` parameter). These are PTXAS-level controls and may not be applicable to TileIR (which uses tileiras, not ptxas directly). Test both:

1. Standard Triton + ACF files
2. TileIR without ACF files

And submit whichever is faster.

---

# 16. Environment Variables Reference

## 16.1 TileIR Backend Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_TILE` | unset | **Required**: Set to "1" to activate TileIR |
| `HELION_BACKEND` | "triton" | Set to "tileir" for Helion integration |
| `TILEIR_ENABLE_APPROX` | "0" | Enable approximate math (fast exp, log, etc.) |
| `TILEIR_ENABLE_FTZ` | "0" | Enable flush-to-zero for denormals |
| `TILE_IR_DISABLE_FMAD` | "0" | Disable FMA fusion (set to "1" to disable) |
| `NVT_TMA_OFFSET_CHECK` | "0" | Enable TMA offset assertion checks |
| `TRITON_TILEIRAS_PATH` | auto-detected | Override path to tileiras binary |
| `TRITON_OVERRIDE_ARCH` | auto-detected | Override GPU architecture (e.g., "sm100") |

## 16.2 Undocumented tileiras Variables

| Variable | Description |
|----------|-------------|
| `TILEIR_ALWAYS_SWIZZLE` | Forces swizzle mode for memory access |
| `TILEIR_PREFER_TMA_FOR_LOAD_STORE` | Prefers TMA for all memory operations |
| `TILEIR_DELAY_TMA_STORE_WAIT` | Delays store waits for compute overlap |

## 16.3 Helion Variables (Unchanged but Relevant)

| Variable | Description |
|----------|-------------|
| `HELION_AUTOTUNE_COMPILE_TIMEOUT` | Per-config compile timeout (recommend 20s for TileIR) |
| `HELION_AUTOTUNE_EFFORT` | none/quick/full |
| `HELION_PRINT_OUTPUT_CODE` | Print generated Triton/TileIR code |
| `HELION_FORCE_AUTOTUNE` | Override provided configs |

---

# 17. Relationship to CUDA Ecosystem

## 17.1 Technology Stack

```
Application Layer:
  [Helion] ──────────── Python DSL, PyTorch syntax
      |
Compiler Frontend:
  [Triton] ──────────── TTIR (Triton IR, MLIR)
      |
      ├── Standard Path ────── TTGIR → LLVM IR → PTX → CUBIN
      |                        (thread-level decomposition)
      |
      └── TileIR Path ──────── CUDA Tile IR → tileiras → CUBIN
                                (tile-level preserved)

Hardware:
  [NVIDIA Blackwell GPU] ─── Tensor Cores, TMA, HBM3e
```

## 17.2 Related Technologies

| Technology | Relationship |
|------------|-------------|
| **CuTe** (CUTLASS 3.0+) | C++ template abstractions for thread/data layouts. Lower-level than Tile IR -- requires explicit layout algebra |
| **CUDA Tile IR** | Virtual ISA for tile-based GPU programming. Higher-level than CuTe -- compiler handles thread mapping |
| **cuTile Python** | NVIDIA's Python DSL targeting Tile IR directly (separate from Triton) |
| **cuTile.jl** | Julia frontend for Tile IR |
| **CUTLASS 4 / DSL** | Python-native interfaces based on CuTe concepts |
| **tileiras** | NVIDIA's Tile IR assembler. Compiles Tile IR bytecode to SASS via LLVM |
| **ptxas** | Standard PTX assembler (used by standard Triton backend) |
| **NVIDIA/cuda-tile** | Open-source Tile IR dialect, Python bindings, bytecode system |

## 17.3 Key Distinction

**CuTe** requires explicit layout algebra and thread coordination at the programming level.
**CUDA Tile IR** lets the compiler handle all of this automatically.
**Triton-to-tile-IR** bridges Triton's tile-based programming model to CUDA Tile IR's tile-based compilation model, creating an end-to-end tile-preserving pipeline.

---

# 18. Sources

- [Triton-to-tile-IR GitHub Repository](https://github.com/triton-lang/Triton-to-tile-IR)
- [Performance Tuning Tips (official)](https://github.com/triton-lang/Triton-to-tile-IR/blob/main/third_party/tileir/PerformanceTuningTips.md)
- [Helion TileIR Backend Documentation](https://github.com/pytorch/helion/blob/main/docs/tileir_backend.md)
- [NVIDIA Blog: Advancing GPU Programming with CUDA Tile IR Backend](https://developer.nvidia.com/blog/advancing-gpu-programming-with-the-cuda-tile-ir-backend-for-openai-triton)
- [NVIDIA Blog: Focus on Your Algorithm -- CUDA Tile Handles the Hardware](https://developer.nvidia.com/blog/focus-on-your-algorithm-nvidia-cuda-tile-handles-the-hardware/)
- [NVIDIA Blog: OpenAI Triton on Blackwell](https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-boosts-ai-performance-and-programmability/)
- [NVIDIA Blog: CUDA 13.1 with CUDA Tile](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/)
- [NVIDIA TileIR Internals (Henry Zhu)](https://maknee.github.io/blog/2026/NVIDIA-TileIR-Internals-from-CuTile-to-MLIR-LLVM-to-SASS/)
- [NVIDIA/cuda-tile Repository](https://github.com/NVIDIA/cuda-tile)
- [NVIDIA Tile IR Documentation](https://docs.nvidia.com/cuda/tile-ir/)
- [Triton Kernel Compilation Stages (PyTorch Blog)](https://pytorch.org/blog/triton-kernel-compilation-stages/)
- [Helion Blog Post](https://pytorch.org/blog/helion/)
- [Helion Performance Tuning Guide](https://github.com/triton-lang/Triton-to-tile-IR/blob/main/HelionPerformanceTuningGuide.md)
- [nvtriton v3.6.0-rc1 Release](https://github.com/triton-lang/Triton-to-tile-IR/releases/tag/v3.6.0-rc1)

---

*Analysis compiled: March 14, 2026*
*Target: NVIDIA B200 (Blackwell sm_100), Helion Hackathon*
*Recommendation: Use TileIR with num_ctas=2 for all dot-heavy kernels (chunk_fwd_h, chunk_fwd_o, recompute_w_u)*
