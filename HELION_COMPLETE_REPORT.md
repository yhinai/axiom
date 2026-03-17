# Helion: Complete Deep Analysis & Comprehensive Report

## A Cutting-Edge Framework for High-Performance GPU Kernel Development

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What is Helion?](#2-what-is-helion)
3. [Why Helion Matters](#3-why-helion-matters)
4. [Architecture & Compilation Pipeline](#4-architecture--compilation-pipeline)
5. [Complete Language API Reference](#5-complete-language-api-reference)
6. [Configuration System](#6-configuration-system)
7. [Autotuner: Architecture & Search Algorithms](#7-autotuner-architecture--search-algorithms)
8. [Performance Optimization Techniques](#8-performance-optimization-techniques)
9. [Advanced Features](#9-advanced-features)
10. [Debug, Profiling & Developer Experience](#10-debug-profiling--developer-experience)
11. [Environment Variables Reference](#11-environment-variables-reference)
12. [Example Kernels Gallery](#12-example-kernels-gallery)
13. [Hackathon Competition Analysis](#13-hackathon-competition-analysis)
14. [Kernel-by-Kernel Deep Analysis](#14-kernel-by-kernel-deep-analysis)
15. [NVIDIA B200 GPU Target Platform](#15-nvidia-b200-gpu-target-platform)
16. [Ecosystem Contribution Opportunities](#16-ecosystem-contribution-opportunities)
17. [Sources & References](#17-sources--references)

---

# 1. Executive Summary

**Helion** is a Python-embedded domain-specific language (DSL) created by Meta's PyTorch Compilers team for authoring high-performance machine learning GPU kernels. It compiles down to Triton (which subsequently compiles to CUDA PTX/CUBIN), positioning itself as **"PyTorch with tiles"** -- the highest-abstraction GPU kernel framework currently available while still achieving near-hand-tuned or better-than-hand-tuned performance.

**Key Metrics:**
- On NVIDIA B200, Helion achieves a geomean speedup of **3.27x** over eager PyTorch
- **1.21x** faster than `torch.compile` (max-autotune)
- **1.85x** faster than hand-written Triton kernels
- Autotuner evaluates hundreds to thousands of configurations (~1520 in one example run)
- BSD-3-Clause open-source license, part of the PyTorch ecosystem

**The name** references nuclear physics: "Helion" = helium-3 nucleus, while "Triton" = hydrogen-3 nucleus -- signifying its close relationship with Triton.

---

# 2. What is Helion?

## 2.1 Core Identity

Helion is a **domain-specific language embedded in Python** for writing GPU kernels. Unlike raw CUDA or even Triton, Helion lets developers express kernel algorithms using familiar PyTorch syntax while a powerful autotuning engine handles all low-level optimization decisions.

## 2.2 Requirements

| Requirement | Version |
|-------------|---------|
| OS | Linux (required) |
| Python | 3.10 - 3.14 |
| PyTorch | 2.9+ |
| Triton | 3.5+ |
| License | BSD-3-Clause |

## 2.3 Installation

```bash
pip install helion
# Or from source:
git clone https://github.com/pytorch/helion
cd helion && pip install -e .
```

## 2.4 The Programming Model

Every Helion kernel has exactly two parts:

1. **Host Code (CPU)**: Standard PyTorch code outside `hl.tile` loops -- allocates output tensors, computes shapes, sets up constants
2. **Device Code (GPU)**: Code inside `hl.tile` loops -- compiled into a single high-performance Triton kernel

```python
import torch
import helion
import helion.language as hl

@helion.kernel()
def my_kernel(x: torch.Tensor) -> torch.Tensor:
    # === HOST CODE (CPU) ===
    m, n = x.size()
    out = torch.empty_like(x)

    # === DEVICE CODE (GPU) ===
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = torch.relu(x[tile_m, tile_n])

    return out
```

---

# 3. Why Helion Matters

## 3.1 The GPU Programming Landscape

| Framework | Abstraction Level | Control | Productivity | Portability |
|-----------|------------------|---------|-------------|-------------|
| CUDA/CuTe-DSL | Lowest | Maximum | Low | None |
| Gluon (OpenAI) | Low | High | Low-Medium | None |
| TLX (Meta) | Low | High | Medium | None |
| **Triton** | Medium | Medium | Medium | Limited |
| ThunderKittens | Medium-High | Medium | Medium | Limited |
| **Helion** | **Highest** | **Medium-High** | **High** | **Multi-platform** |
| PyTorch | Highest | Lowest | Highest | Full |

## 3.2 The False Dichotomy Helion Solves

Before Helion, developers faced a painful tradeoff:

- **CUDA**: Maximum performance but enormous development effort, no portability
- **Triton**: Better abstraction but still requires manual indexing, stride calculation, masking, search space definition
- **PyTorch**: Maximum productivity but no fine-grained kernel control

Helion eliminates this tradeoff by providing PyTorch-level productivity with near-CUDA-level performance through automated optimization.

## 3.3 Seven Key Automations

1. **Tensor Indexing**: Automatically calculates strides and indices; autotunes among pointer, block_ptr, and TensorDescriptor access methods per operation
2. **Masking**: Implicit masking that is optimized away when provably unnecessary
3. **Grid Configuration**: Automatic grid size determination and PID-to-tile mapping
4. **Search Space Generation**: No manual search space definition -- automatically derived from kernel structure
5. **Argument Management**: Tensor sizes/strides automated; closure variables lifted into kernel parameters
6. **Reduction Optimization**: Automatic conversion of large reductions to looped implementations
7. **Low-Level Optimizations**: PID swizzling, loop reordering, persistent kernels, warp specialization, unrolling -- all explored by the autotuner

## 3.4 Performance Results

On NVIDIA B200 (Blackwell):
- **3.27x** geomean speedup over eager PyTorch
- **1.21x** over torch.compile (max-autotune)
- **1.85x** over hand-written Triton

This means Helion's autotuner often discovers optimizations that even expert Triton programmers miss.

---

# 4. Architecture & Compilation Pipeline

## 4.1 High-Level Pipeline

```
Python Source with @helion.kernel()
        |
        v
[1] Decorator Processing
    - Creates Kernel object
    - Validates signatures
    - Extracts constexpr annotations
        |
        v
[2] Specialization
    - Generates specialization keys from:
      tensor properties (dtype, device, shape, stride),
      SymInts, custom extractors, dataclasses,
      function closures
    - static_shapes=True generates separate keys
      per unique shape/stride signature
        |
        v
[3] Binding
    - Creates BoundKernel with argument normalization
    - Fake tensor creation for tracing
        |
        v
[4] Type Propagation
    - AST-level analysis determines types
    - Resolves tile sizes
    - Builds ConfigSpec (search space)
        |
        v
[5] Device IR Generation
    - Traces device code via PythonKeyTracer
    - Builds FX graph
    - PyTorch operators mapped through
      TorchInductor decomposition tables
        |
        v
[6] Code Generation
    - Backend emits target code:
      * Triton (@triton.jit with tl.* ops)
      * Pallas (experimental)
      * CUTE (experimental)
    - Proper indexing, masking, grid configuration
        |
        v
[7] Autotuning (if no config provided)
    - Explores ConfigSpec search space
    - 7 search algorithms available
    - Ephemeral Triton caching to avoid pollution
        |
        v
[8] Execution & Caching
    - Compiled kernel launched
    - Results cached by specialization key
    - Subsequent calls skip compilation
```

## 4.2 TorchInductor Integration

Helion leverages TorchInductor (PyTorch 2's core compiler component) to automatically map PyTorch operations to low-level Triton implementations. This means any PyTorch operator that TorchInductor supports can be used inside Helion kernels:

- **Pointwise**: add, sub, mul, div, sigmoid, relu, exp, log, abs, pow, sqrt, clamp
- **Reductions**: sum, mean, amax, amin, softmax
- **Views**: reshape, permute, transpose, expand, contiguous
- **Matrix Multiply**: torch.addmm, torch.baddbmm, torch.matmul, @ operator
- **Comparisons**: gt, lt, ge, le, eq, ne, where
- **Math**: minimum, maximum, floor, ceil, round

---

# 5. Complete Language API Reference

## 5.1 Loop Constructs

### `hl.tile(begin_or_end, end_or_none=None, /, block_size=None)`
The fundamental parallelization construct. Subdivides the iteration space into tiles that execute in parallel on the GPU.

- **Returns**: `Iterator[Tile]` (1D) or `Iterator[Sequence[Tile]]` (multi-D)
- **Top-level**: Becomes the GPU grid (determines how many thread blocks launch)
- **Nested**: Becomes a loop within the kernel
- **block_size=None**: Autotuned by the search engine
- **block_size=int**: Fixed tile size

**Tile object properties:**
- `.begin` -- start index of this tile
- `.end` -- end index of this tile
- `.block_size` -- size of this tile
- `.index` -- full index tensor (equivalent to `arange(begin, end)`)
- `.id` -- tile ID within the grid

**Usage patterns:**
```python
# 1D tiling
for tile_m in hl.tile(m):
    out[tile_m] = x[tile_m] + 1.0

# 2D tiling (generates 2D grid)
for tile_m, tile_n in hl.tile([m, n]):
    out[tile_m, tile_n] = x[tile_m, tile_n]

# Nested tiling (outer = grid, inner = loop)
for tile_m in hl.tile(m):
    acc = hl.zeros([tile_m], dtype=torch.float32)
    for tile_n in hl.tile(n):
        acc += torch.sum(x[tile_m, tile_n], dim=1)

# Fixed block size
for tile_m in hl.tile(m, block_size=128):
    ...

# With begin/end range
for tile_m in hl.tile(start, end):
    ...
```

### `hl.grid(begin_or_end, end_or_none=None, /, step=None)`
Iterates over **scalar** integer indices (`torch.SymInt`), unlike `tile` which gives `Tile` objects that load slices. Same grid/loop semantics as `tile` but element-at-a-time. Serves as the device-side replacement for Python's `builtins.range`.

### `hl.static_range(begin_or_end, end_or_none=None, /, step=1)`
Compile-time unrolled range over constant integers. Must be small (<=8 iterations by default) to avoid compile-time explosion. Useful for small, fixed-size loops (e.g., filter width in convolution).

## 5.2 Tensor Creation (Device-Local)

### `hl.zeros(shape, dtype=torch.float32, device=None)`
Creates a device tensor filled with zeros. Used for accumulators.
```python
acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
```

### `hl.full(shape, value, dtype=torch.float32, device=None)`
Creates a device tensor filled with `value`.
```python
mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
```

### `hl.arange(*args, dtype=None, device=None, **kwargs)`
Same semantics as `torch.arange()` but defaults to kernel's device and index dtype.
```python
idx = hl.arange(tile_t.block_size)
mask = idx[:, None] >= idx[None, :]  # causal mask
```

## 5.3 Matrix Operations

### `hl.dot(mat1, mat2, acc=None, out_dtype=None)`
Matrix multiplication leveraging tensor cores.

- **Supported dtypes**: float16, bfloat16, float32, int8, float8_e4m3fn, float8_e5m2
- **acc**: Optional accumulator (float16, float32, int32). If provided, returns `acc + (mat1 @ mat2)`
- **out_dtype**: Controls output type of the multiplication prior to accumulation
- **Hardware constraints**: Enforces minimum block sizes per device
- **Batched**: Supports 2D and 3D tensors (batch dim constrained to block_size=1 for 3D)

```python
# Basic matmul
result = hl.dot(q_scaled, k_scaled.T)

# With accumulator
acc = hl.dot(a, b, acc=acc, out_dtype=torch.float32)

# FP8 matmul
result = hl.dot(fp8_a, fp8_b, out_dtype=torch.float32)
```

### `hl.dot_scaled(mat1, mat1_scale, mat1_format, mat2, mat2_scale, mat2_format, acc=None, out_dtype=None)`
Block-scaled matrix multiplication via Triton's `tl.dot_scaled`.
- **Formats**: "e2m1", "e4m3", "e5m2", "bf16", "fp16"
- **Scale**: e8m0 format (uint8): `value = 2^(byte - 127)`
- **Constraints**: K dimension >= 32, 2D tensors only

## 5.4 Reduction Operations

### `hl.reduce(combine_fn, input_tensor, dim=None, other=0, keep_dims=False)`
User-defined reduction with custom combine function (must be associative and commutative).
- Standard PyTorch reductions (sum, mean, amax) work directly without this function
- Supports single tensor or tuple of tensors
- `other`: fill value for masked/padded elements
- CUTE backend supports builtin reductions and argmax/argmin patterns

## 5.5 Scan Operations

### `hl.associative_scan(combine_fn, input_tensor, dim, reverse=False)`
Prefix scan (cumulative operation) preserving input shape. `combine_fn` must be associative.

### `hl.cumsum(input_tensor, dim, reverse=False)`
Cumulative sum. Equivalent to `hl.associative_scan(torch.add, ...)`.

### `hl.cumprod(input_tensor, dim, reverse=False)`
Cumulative product. Equivalent to `hl.associative_scan(torch.mul, ...)`.

## 5.6 Memory Operations

### `hl.load(tensor, index, extra_mask=None, eviction_policy=None)`
Explicit load from tensor using list of indices.
- `extra_mask`: Additional masking beyond automatic tile bounds
- `eviction_policy`: Cache hint forwarded to Triton `tl.load`

```python
x_val = hl.load(x_pad, [batch_idx, channel_tile, seq_tile.index + j])
```

### `hl.store(tensor, index, value, extra_mask=None)`
Explicit store to tensor using list of indices.

## 5.7 Atomic Operations

| Function | Description |
|----------|-------------|
| `hl.atomic_add(tensor, index, value)` | Atomic addition |
| `hl.atomic_and(tensor, index, value)` | Atomic bitwise AND |
| `hl.atomic_or(tensor, index, value)` | Atomic bitwise OR |
| `hl.atomic_xor(tensor, index, value)` | Atomic bitwise XOR |
| `hl.atomic_max(tensor, index, value)` | Atomic maximum |
| `hl.atomic_min(tensor, index, value)` | Atomic minimum |
| `hl.atomic_xchg(tensor, index, value)` | Atomic exchange |
| `hl.atomic_cas(tensor, index, cmp, val)` | Atomic compare-and-swap |

## 5.8 Compile-Time Specialization

### `hl.specialize(value)`
Turns dynamic shapes into compile-time constants. Generates a separate compiled kernel per distinct value.

```python
K = hl.specialize(K)   # Bake K into the kernel
V = hl.specialize(V)   # Bake V into the kernel
channels = hl.specialize(tensor.size(1))
height, width = hl.specialize(tensor.shape[-2:])
```

### `hl.constexpr`
Type annotation or call-site wrapper for compile-time constants.
```python
def kernel(x: torch.Tensor, eps: hl.constexpr) -> torch.Tensor:
    ...
# or
kernel(x, hl.constexpr(1e-6))
```

## 5.9 Tunable Parameters

### `hl.register_block_size(size)`
Registers a named block size for autotuning. Returns an opaque handle used with `hl.tile`.

```python
block_size_m = hl.register_block_size(m)
for tile_m in hl.tile(m, block_size=block_size_m):
    ...
```

### `hl.register_tunable(name, values)`
Registers user-defined tunable parameters that the autotuner will explore.

## 5.10 Synchronization Primitives

### `hl.barrier()`
Grid-wide barrier separating top-level `hl.tile`/`hl.grid` loops. Host-level only (no device code emitted). Forces `persistent_blocked` PID type.

### `hl.wait(signal_pad, index=None, signal=1, update=None, scope="gpu", hasSubsequentMemAccess=True)`
Spins on global memory barriers until signal value is observed. For inter-CTA synchronization.

### `hl.signal(signal_pad, index=None, signal=1, wait_for=None, scope="gpu", hasPreviousMemAccess=True)`
Sets global memory barriers for inter-CTA communication.

## 5.11 Escape Hatches

### `hl.inline_triton(triton_source, args, output_like)`
Embeds raw Triton code snippets inside Helion kernels.
```python
result = hl.inline_triton(
    "tl.exp({0}) + {1}",
    [x_tile, bias_tile],
    output_like=x_tile
)
```

### `hl.triton_kernel(triton_source_or_fn, args, output_like)`
Defines and calls a `@triton.jit` function from device code. Accepts source strings or Python function objects.

### `hl.inline_asm_elementwise(...)`
Inline PTX assembly escape hatch for maximum control.

**Competition Rule**: These escape hatches are allowed but must constitute <=30% of kernel LOC.

## 5.12 Debug Operations

### `hl.device_print(...)`
Print from inside device code (requires `TRITON_INTERPRET=1`).

### `hl.breakpoint()`
Set breakpoint inside device code (requires `TRITON_INTERPRET=1`).

## 5.13 Other APIs

| API | Description |
|-----|-------------|
| `hl.StackTensor` | Stack tensor type for multi-buffer patterns |
| `hl.stacktensor_like(tensor)` | Create StackTensor matching existing tensor |
| `hl.join(tensors, dim)` | Join tensors along dimension |
| `hl.split(tensor, sizes, dim)` | Split tensor along dimension |
| `hl.subscript(tensor, index)` | Subscript operation |
| `hl.tile_begin(tile)` | Get tile start index |
| `hl.tile_end(tile)` | Get tile end index |
| `hl.tile_block_size(tile)` | Get tile size |
| `hl.tile_id(tile)` | Get tile identifier |
| `hl.tile_index(tile)` | Get tile index tensor |
| `hl.rand(shape, dtype)` | Random number generation |
| `hl.randint(low, high, shape)` | Random integer generation |

---

# 6. Configuration System

## 6.1 Config vs Settings

Helion separates two parameter namespaces:

- **Config** (`helion.Config`): Controls GPU execution -- performance-focused parameters explored by the autotuner
- **Settings** (`helion.Settings`): Controls compilation behavior -- developer-facing knobs for debugging, effort, backends

## 6.2 Config Parameters (Complete Reference)

| Parameter | Type | Valid Values | Description |
|-----------|------|-------------|-------------|
| `block_sizes` | `list[int]` | Powers of 2 | Tile dimensions for each `hl.tile` call |
| `num_warps` | `int` | 1, 2, 4, 8, 16, 32 | Warps per thread block |
| `num_stages` | `int` | 1-8 | Software pipelining depth |
| `indexing` | `str` or `list[str]` | "pointer", "block_ptr", "tensor_descriptor" | Memory access strategy |
| `pid_type` | `str` | "flat", "xyz", "persistent_blocked", "persistent_interleaved" | PID-to-tile mapping |
| `loop_orders` | `list[list[int]]` | Permutations | Iteration order for multi-D tiles |
| `l2_groupings` | `list[int]` | Powers of 2, 1=disabled | L2 cache locality optimization |
| `flatten_loops` | `list[bool]` | True/False | Merge multi-D into single dim |
| `reduction_loops` | `list[int\|None]` | Powers of 2 | Persistent vs looped reduction |
| `elements_per_thread` | `list[int]` or `int` | Positive ints | Work per thread |
| `range_unroll_factors` | `list[int]` | 0-4 | Loop unrolling (0 = no unroll) |
| `range_warp_specializes` | `list[bool\|None]` | None/False/True | Warp specialization (Blackwell+) |
| `range_num_stages` | `list[int]` | 0-4 | Pipelining per range loop |
| `range_multi_buffers` | `list[bool\|None]` | None/False/True | Multi-buffering |
| `range_flattens` | `list[bool\|None]` | None/False/True | Range flattening |
| `static_ranges` | `list[bool]` | True/False | Static vs dynamic ranges |
| `load_eviction_policies` | `list[str]` | "", "first", "last" | Cache eviction hints |
| `num_sm_multiplier` | `int` | 1, 2, 4, 8 | SM multiplier for persistent kernels |
| `maxnreg` | `int\|None` | 32, 64, 128, 256, None | Max registers per thread |
| `advanced_controls_file` | `str\|None` | File path | PTXAS ACF file for advanced tuning |
| `**kwargs` | object | Any | User-defined tunable parameters |

### Config Methods

```python
config = helion.Config(block_sizes=[64, 64], num_warps=4)
config.to_json()           # Serialize to JSON string
config.save("config.json") # Save to file (atomic write)

config = helion.Config.from_json(json_str)  # Deserialize
config = helion.Config.from_dict(dict_obj)  # From dictionary
config = helion.Config.load("config.json")  # Load from file

minimized = config.minimize(config_spec)    # Remove default values
```

## 6.3 Settings Parameters (Complete Reference)

| Setting | Env Variable | Default | Description |
|---------|-------------|---------|-------------|
| `backend` | `HELION_BACKEND` | auto | Code gen backend (triton/pallas/cute) |
| `static_shapes` | `HELION_STATIC_SHAPES` | True | Specialize per shape for max perf |
| `dot_precision` | `TRITON_F32_DEFAULT` | varies | Dot precision: tf32/tf32x3/ieee |
| `fast_math` | `HELION_FAST_MATH` | False | Enable fast math approximations |
| `index_dtype` | `HELION_INDEX_DTYPE` | auto | dtype for index variables |
| `autotune_effort` | `HELION_AUTOTUNE_EFFORT` | "full" | none/quick/full |
| `force_autotune` | `HELION_FORCE_AUTOTUNE` | False | Override provided configs |
| `autotune_random_seed` | `HELION_AUTOTUNE_RANDOM_SEED` | auto | Reproducible search |
| `autotune_compile_timeout` | `HELION_AUTOTUNE_COMPILE_TIMEOUT` | 60s | Per-config compile timeout |
| `autotune_precompile` | `HELION_AUTOTUNE_PRECOMPILE` | "fork" | Precompile mode (fork/spawn/None) |
| `autotune_precompile_jobs` | `HELION_AUTOTUNE_PRECOMPILE_JOBS` | None | Parallel precompile workers |
| `autotune_accuracy_check` | `HELION_AUTOTUNE_ACCURACY_CHECK` | True | Verify accuracy during tuning |
| `autotune_search_acf` | `HELION_AUTOTUNE_SEARCH_ACF` | [] | ACF files to explore |
| `autotune_progress_bar` | `HELION_AUTOTUNE_PROGRESS_BAR` | True | Show progress bar |
| `autotune_max_generations` | `HELION_AUTOTUNE_MAX_GENERATIONS` | None | Override max generations |
| `autotune_ignore_errors` | `HELION_AUTOTUNE_IGNORE_ERRORS` | False | Skip erroring configs |
| `autotune_adaptive_timeout` | `HELION_AUTOTUNE_ADAPTIVE_TIMEOUT` | True | Adaptive compile timeout |
| `autotune_cache` | `HELION_AUTOTUNE_CACHE` | "LocalAutotuneCache" | Cache implementation |
| `autotune_best_available_max_configs` | env var | 20 | Max configs for best-available |
| `autotune_best_available_max_cache_scan` | env var | 500 | Max cache entries to scan |
| `print_output_code` | `HELION_PRINT_OUTPUT_CODE` | False | Print generated Triton |
| `print_repro` | `HELION_PRINT_REPRO` | False | Generate repro script |
| `output_origin_lines` | `HELION_OUTPUT_ORIGIN_LINES` | True | Source line comments in output |
| `ignore_warnings` | `HELION_IGNORE_WARNINGS` | [] | Suppress specific warnings |
| `allow_warp_specialize` | `HELION_ALLOW_WARP_SPECIALIZE` | True | Allow warp specialization |
| `debug_dtype_asserts` | `HELION_DEBUG_DTYPE_ASSERTS` | False | Runtime dtype assertions |
| `persistent_reserved_sms` | `HELION_PERSISTENT_RESERVED_SMS` | 0 | Reserved SMs |
| `autotune_force_persistent` | `HELION_AUTOTUNE_FORCE_PERSISTENT` | False | Persistent-only search |
| `autotune_log` | -- | None | Log file path |
| `autotune_log_level` | -- | auto | Log verbosity |
| `autotune_rebenchmark_threshold` | `HELION_REBENCHMARK_THRESHOLD` | None | Rebenchmark threshold |

---

# 7. Autotuner: Architecture & Search Algorithms

## 7.1 Overview

Helion's autotuner is one of its most sophisticated components. It systematically explores the multi-dimensional configuration space to find optimal kernel parameters for the target hardware.

## 7.2 Search Algorithms (7 Available)

| Algorithm | Type | Description |
|-----------|------|-------------|
| **LFBOTreeSearch** | Surrogate-Guided | *Default*. Random Forest classifier as surrogate model with tree-guided neighbor generation |
| **LFBOPatternSearch** | Surrogate + Local | Random Forest with pattern-search-style perturbations |
| **DESurrogateHybrid** | Surrogate + Evolutionary | Differential Evolution with surrogate model filtering |
| **DifferentialEvolutionSearch** | Evolutionary | Standard DE: population=40, generations=20, crossover=0.8 |
| **PatternSearch** | Local Search | Single-parameter perturbations around current best |
| **RandomSearch** | Global | Baseline with configurable count (100-1000) |
| **FiniteSearch** | Exhaustive | Brute-force over predefined config list |

## 7.3 Effort Profiles

### "none" -- Instant startup
- All algorithms disabled
- Uses default config (slow kernel but zero autotuning cost)
- Best for: development iteration, correctness testing

### "quick" -- ~3 seconds, ~32 configs
- PatternSearch: population=30, copies=2, generations=5, strategy="from_default"
- DifferentialEvolution: population=20, generations=8, strategy="from_default"
- RandomSearch: count=100
- rebenchmark_threshold=0.9
- Best for: initial exploration, CI/CD

### "full" -- ~10+ minutes, 1000+ configs
- PatternSearch: population=100, copies=5, generations=20, strategy="from_random"
- DifferentialEvolution: population=40, generations=40, strategy="from_random"
- RandomSearch: count=1000
- rebenchmark_threshold=1.5
- Best for: production deployment, competition

## 7.4 Config Fragments (Search Space Building Blocks)

The autotuner represents each tunable parameter as a "fragment" -- a discrete or continuous space to search:

| Fragment Type | Description | Example |
|---------------|-------------|---------|
| `BooleanFragment` | True/False | flatten_loops |
| `EnumFragment` | Discrete set | {None, False, True} for optional bools |
| `IntegerFragment` | Integer range | 0-4 for unroll factors |
| `PowerOfTwoFragment` | Powers of 2 in [min, max] | block_sizes: 16-1024 |
| `ListOf` | Applies fragment to list elements | block_sizes per tile dimension |
| `PermutationFragment` | All permutations of indices | loop_orders |

## 7.5 ConfigSpec (Automatic Search Space)

The `ConfigSpec` class automatically assembles the complete search space from the kernel's structure:

- **BlockSizeSpec**: Powers of 2 with `size_hint`, `min_size`, `max_size` (respects hardware minimums for dot operations)
- **LoopOrderSpec**: Permutations of dimension indices
- **L2GroupingSpec**: Powers of 2 (1=disabled)
- **ReductionLoopSpec**: Powers of 2 with backend max_reduction_threads limit
- **RangeUnrollFactorSpec**: Integer 0-4
- **RangeWarpSpecializeSpec**: Enum {None, False, True}
- **RangeNumStagesSpec**: Integer 0-4

The spec generates a SHA-256 `structural_fingerprint` for cache key matching.

## 7.6 Caching

| Cache Class | Description |
|-------------|-------------|
| `LocalAutotuneCache` | Default: per-device Triton cache with config serialization |
| `StrictLocalAutotuneCache` | Stricter matching for reproducibility |
| `AOTAutotuneCache` | Ahead-of-time compilation cache |

## 7.7 Autotuning Workflow

```python
# Method 1: Automatic (no config specified)
@helion.kernel()
def my_kernel(x):  # First call triggers ~10 min autotuning
    ...

# Method 2: Multiple configs (lightweight selection)
@helion.kernel(configs=[
    helion.Config(block_sizes=[32, 32], num_warps=4),
    helion.Config(block_sizes=[64, 64], num_warps=8),
])
def my_kernel(x):  # Picks fastest of the provided configs
    ...

# Method 3: Fixed config (zero autotuning)
@helion.kernel(config=helion.Config(block_sizes=[64, 64], num_warps=4))
def my_kernel(x):  # Uses exactly this config
    ...

# Method 4: Programmatic autotuning
best = my_kernel.autotune(args=(x, y), force=True)
best.save("best_config.json")
```

---

# 8. Performance Optimization Techniques

## 8.1 Indexing Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `"pointer"` | Traditional pointer arithmetic | Irregular access patterns |
| `"block_ptr"` | Triton block pointers | Regular, strided access (most common) |
| `"tensor_descriptor"` | TMA-based (Hopper+) | Large, aligned tensor transfers |

The autotuner selects per-operation. Block pointers are often best for regular matmuls; pointer arithmetic for irregular/jagged access.

## 8.2 PID Type Strategies

| PID Type | Description | Use Case |
|----------|-------------|----------|
| `"flat"` | Linear PID mapping | Simple kernels |
| `"xyz"` | CUDA xyz grid dims (max 3D) | Multi-dimensional grids |
| `"persistent_blocked"` | Persistent kernel, blocked PIDs | High-occupancy, barriers |
| `"persistent_interleaved"` | Persistent kernel, interleaved PIDs | Load balancing |

## 8.3 L2 Cache Optimization

`l2_groupings` reorders PIDs so that nearby thread blocks access nearby memory, improving L2 cache hit rates. Values are powers of 2 (1=disabled). Critical for matmul-heavy kernels.

## 8.4 Loop Optimization

| Technique | Config Parameter | Description |
|-----------|-----------------|-------------|
| Loop reordering | `loop_orders` | Permute iteration dimensions for better locality |
| Loop flattening | `flatten_loops` | Merge multi-D iterations into single dimension |
| Unrolling | `range_unroll_factors` | 0-4x unrolling for inner loops |
| Pipelining | `range_num_stages` | Software pipelining for memory-bound loops |
| Multi-buffering | `range_multi_buffers` | Double-buffer loads to hide latency |
| Static ranges | `static_ranges` | Compile-time unroll for known-small loops |

## 8.5 Persistent Kernels

Persistent kernels keep thread blocks alive across multiple tiles, avoiding re-launch overhead:
- Enable via `pid_type="persistent_blocked"` or `"persistent_interleaved"`
- `num_sm_multiplier`: 1, 2, 4, or 8x SM count
- `persistent_reserved_sms`: Reserve SMs for concurrent work
- `autotune_force_persistent=True` restricts to persistent variants

## 8.6 Register Pressure

`maxnreg` caps registers at 32, 64, 128, or 256 per thread. Lower register counts allow more concurrent warps (higher occupancy) at the cost of potential register spills.

## 8.7 ACF Files (PTXAS Advanced Controls)

Pre-tuned PTXAS advanced control files provide low-level compiler hints. Available on B200 at `/opt/booster_pack/`:

```
/opt/booster_pack/
├── causal_conv_*.acf           (3 files)
├── chunk_fwd_h_*.acf           (2 files)
├── chunk_fwd_o_*.acf           (7 files)
├── fp8_group_quant_*.acf       (7 files)
└── recompute_w_u_fwd_*.acf     (5 files)
```

Usage:
```python
# During autotuning: search ACF files
@helion.kernel(autotune_search_acf=[
    "/opt/booster_pack/fp8_group_quant_0.acf",
    "/opt/booster_pack/fp8_group_quant_1.acf",
])
def kernel(...):

# In production: hardcode best ACF
helion.Config(
    block_sizes=[64],
    num_warps=4,
    advanced_controls_file="/opt/booster_pack/fp8_group_quant_3.acf"
)
```

---

# 9. Advanced Features

## 9.1 Warp Specialization (Blackwell+ Only)

Warp specialization assigns different warps to different roles within a thread block (e.g., data loading vs computation). Available on NVIDIA Blackwell (compute capability >= 10.0).

- Config: `range_warp_specializes: [True]`
- Setting: `allow_warp_specialize=True` (default)
- The autotuner explores this as an `EnumFragment` with values {None, False, True}

## 9.2 TileIR Backend

NVIDIA's experimental Triton backend that bypasses LLVM and compiles directly to CUBIN via `tileiras`. Targets Blackwell compute capability 10.x and 12.x.

```bash
export ENABLE_TILE=1
export HELION_BACKEND=tileir
# Requires: pip install triton-tileir  (from github.com/triton-lang/Triton-to-tile-IR)
```

TileIR configs include additional parameters: `num_ctas`, `occupancy`.

## 9.3 Paged Attention in Helion

Helion has been used to implement vLLM's core Paged Attention kernel (experimental, PR#27293). Key innovations:
- "Q blocks" parallelization strategy
- Query Group Attention (QGA) optimization
- Autotuner explores both algorithmic and low-level parameters
- Same code runs on NVIDIA, AMD, Intel, and custom accelerators

## 9.4 Signal/Wait Inter-CTA Communication

For advanced algorithms requiring communication between thread blocks:

```python
# Producer CTA
hl.signal(signal_pad, index=[block_id], signal=1)

# Consumer CTA
hl.wait(signal_pad, index=[block_id], signal=1)
```

Supports `scope="sys"` (system-wide, for multi-GPU) or `scope="gpu"` (single GPU).

## 9.5 Templating with Closures

Helion supports lambda epilogues that capture additional arguments:

```python
@helion.kernel()
def matmul(x, y, epilogue=lambda acc, tile: acc):
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = epilogue(acc, (tile_m, tile_n))

# Usage: fused matmul + ReLU + bias
result = matmul(x, y, lambda acc, tile: torch.relu(acc + bias[tile[1]]))
```

## 9.6 Autograd Integration

Helion kernels can be wrapped in `torch.autograd.Function` for forward/backward:

```python
class MyKernelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        result = my_kernel(x, y)
        ctx.save_for_backward(x, y)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x = my_kernel_backward_x(grad_output, y)
        grad_y = my_kernel_backward_y(grad_output, x)
        return grad_x, grad_y
```

---

# 10. Debug, Profiling & Developer Experience

## 10.1 Viewing Generated Code

```bash
# Environment variable
HELION_PRINT_OUTPUT_CODE=1 python my_kernel.py

# Or in decorator
@helion.kernel(print_output_code=True)
```

Prints the generated Triton source code to stderr, showing exactly what `@triton.jit` kernel was produced.

## 10.2 Generating Reproducible Test Scripts

```bash
HELION_PRINT_REPRO=1 python my_kernel.py
# Or
@helion.kernel(print_repro=True)
```

Generates a minimal standalone script that reproduces the kernel execution -- invaluable for bug reports.

## 10.3 Eager/Interpret Mode

```bash
HELION_INTERPRET=1 python my_kernel.py
```

Runs kernel logic on CPU without compilation. Useful for debugging logic errors.

```bash
TRITON_INTERPRET=1 python my_kernel.py
```

Runs through Triton's CPU interpreter, enabling `hl.device_print()` and `hl.breakpoint()`.

## 10.4 Logging

```bash
HELION_LOGS=all           # INFO level for all modules
HELION_LOGS=+all          # DEBUG level for all modules
HELION_LOGS=+helion.runtime.kernel  # Specific module debug
```

## 10.5 Runtime Assertions

```python
@helion.kernel(debug_dtype_asserts=True)
```

Adds runtime dtype assertion checks in generated code -- useful for catching type mismatch bugs.

## 10.6 Source Line Comments

`HELION_OUTPUT_ORIGIN_LINES=1` (default) embeds comments in generated Triton code mapping back to Helion source lines.

## 10.7 Profiling with Nsight Compute

Via the hackathon infrastructure:
```bash
popcorn submit submission.py --gpu B200_Nebius --leaderboard <name> --mode profile
```

---

# 11. Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `HELION_AUTOTUNE_EFFORT` | "full" | Effort: none/quick/full |
| `HELION_PRINT_OUTPUT_CODE` | 0 | Print generated Triton code |
| `HELION_PRINT_REPRO` | 0 | Generate reproducible test script |
| `HELION_FORCE_AUTOTUNE` | 0 | Override provided configs |
| `HELION_INTERPRET` | 0 | Eager mode execution (CPU) |
| `TRITON_INTERPRET` | 0 | Triton CPU interpreter |
| `HELION_BACKEND` | auto | Backend: triton/pallas/cute/tileir |
| `HELION_STATIC_SHAPES` | 1 | Static shape specialization |
| `HELION_FAST_MATH` | 0 | Fast math approximations |
| `HELION_INDEX_DTYPE` | auto | Index variable dtype |
| `HELION_LOGS` | -- | Logging: all, +all, +module.name |
| `HELION_PERSISTENT_RESERVED_SMS` | 0 | Reserved SMs for persistent kernels |
| `HELION_AUTOTUNE_FORCE_PERSISTENT` | 0 | Restrict to persistent kernels |
| `HELION_AUTOTUNE_COMPILE_TIMEOUT` | 60 | Per-config compile timeout (seconds) |
| `HELION_AUTOTUNE_PRECOMPILE` | "fork" | Precompile mode |
| `HELION_AUTOTUNE_PRECOMPILE_JOBS` | None | Parallel precompile workers |
| `HELION_AUTOTUNE_RANDOM_SEED` | auto | Reproducible search seed |
| `HELION_AUTOTUNE_ACCURACY_CHECK` | 1 | Verify accuracy during tuning |
| `HELION_REBENCHMARK_THRESHOLD` | None | Rebenchmark threshold |
| `HELION_AUTOTUNE_SEARCH_ACF` | [] | ACF files for search |
| `HELION_AUTOTUNE_PROGRESS_BAR` | 1 | Show progress bar during tuning |
| `HELION_AUTOTUNE_MAX_GENERATIONS` | None | Override max generations |
| `HELION_AUTOTUNE_IGNORE_ERRORS` | 0 | Skip erroring configs |
| `HELION_AUTOTUNE_ADAPTIVE_TIMEOUT` | 1 | Adaptive compile timeout |
| `HELION_OUTPUT_ORIGIN_LINES` | 1 | Source line comments in output |
| `HELION_ALLOW_WARP_SPECIALIZE` | 1 | Allow warp specialization |
| `HELION_DEBUG_DTYPE_ASSERTS` | 0 | Runtime dtype assertion checks |
| `HELION_AUTOTUNE_CACHE` | "LocalAutotuneCache" | Cache class |
| `HELION_BEST_AVAILABLE_MAX_CONFIGS` | 20 | Max configs for best-available |
| `HELION_BEST_AVAILABLE_MAX_CACHE_SCAN` | 500 | Max cache entries to scan |
| `HELION_AUTOTUNE_CONFIG_OVERRIDES` | -- | Config overrides |
| `TRITON_F32_DEFAULT` | varies | Dot precision (tf32/tf32x3/ieee) |
| `ENABLE_TILE` | 0 | Enable TileIR backend |

---

# 12. Example Kernels Gallery

Helion ships with **55+ example kernels** in `examples/`:

| Category | Examples |
|----------|----------|
| **Basic** | `add.py`, `exp.py`, `concatenate.py` |
| **Normalization** | `rms_norm.py`, `layer_norm.py`, `batch_softmax.py` |
| **Attention** | `attention.py` (FlashAttention), `fp8_attention.py`, `flex_attention.py`, `blackwell_attention.py` |
| **MatMul** | `matmul.py`, `bmm.py`, `broadcast_matmul.py`, `matmul_split_k.py`, `fp8_gemm.py`, `bf16xint16_gemm.py`, `int4_gemm.py`, `nvfp4_gemm.py`, `grouped_gemm.py`, `moe_matmul_ogs.py` |
| **Fused Ops** | `matmul_layernorm.py`, `geglu.py`, `swiglu.py`, `fused_linear_jsd.py`, `squeeze_and_excitation_net.py` |
| **SSM/RNN** | `gdn_fwd_h.py` (Gated DeltaNet), `mamba2_chunk_scan.py`, `mamba2_chunk_state.py` |
| **Jagged** | `jagged_dense_bmm.py`, `jagged_dense_add.py`, `jagged_softmax.py`, `jagged_sum.py`, `jagged_mean.py`, `jagged_layer_norm.py`, `jagged_hstu_attn.py` |
| **Loss** | `cross_entropy.py`, `kl_div.py`, `jsd.py`, `grpo_loss.py` |
| **Other** | `embedding.py`, `softmax.py`, `long_sum.py`, `low_mem_dropout.py`, `segment_reduction.py`, `welford.py`, `outer_product.py` |
| **Advanced** | `split_k_barrier.py` (barrier sync), `aot_compile_example.py`, `aot_example.py`, `distributed/` |

### Key Pattern: Matrix Multiplication

```python
@helion.kernel(static_shapes=True)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out
```

### Key Pattern: FlashAttention (Online Softmax)

```python
@helion.kernel()
def flash_attention(q, k, v):
    batch, seq_len, dim = q.size()
    out = torch.empty_like(q)
    for tile_b, tile_m in hl.tile([batch, seq_len]):
        mi = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        li = hl.zeros([tile_b, tile_m], dtype=torch.float32)
        acc = hl.zeros([tile_b, tile_m, dim], dtype=torch.float32)
        for tile_n in hl.tile(seq_len):
            qk = torch.bmm(q[tile_b, tile_m, :], k[tile_b, tile_n, :].transpose(-1, -2))
            mi_new = torch.maximum(mi, torch.amax(qk, dim=-1))
            alpha = torch.exp(mi - mi_new)
            p = torch.exp(qk - mi_new[:, :, None])
            acc = acc * alpha[:, :, None] + torch.bmm(p, v[tile_b, tile_n, :])
            li = li * alpha + p.sum(dim=-1)
            mi = mi_new
        out[tile_b, tile_m, :] = acc / li[:, :, None]
    return out
```

### Key Pattern: Softmax (Two-Pass)

```python
@helion.kernel()
def softmax_two_pass(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(values - mi_next[:, None]).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out
```

---

# 13. Hackathon Competition Analysis

## 13.1 Event Details

| Attribute | Value |
|-----------|-------|
| **Name** | PyTorch Helion Hackathon |
| **Date** | March 14-15, 2026 (Sat 4pm - Sun 5am PT) |
| **Location** | San Francisco, CA (in-person only) |
| **Co-hosts** | PyTorch, Cerebral Valley |
| **Partners** | Meta, NVIDIA, Nebius, GPU MODE |
| **GPU** | B200_Nebius (NVIDIA B200) |
| **Team Size** | Max 4 people |

## 13.2 Prizes

| Place | Prize |
|-------|-------|
| 1st | NVIDIA DGX Spark + GTC 2026 Pass |
| 2nd | NVIDIA RTX 5090 + GTC 2026 Pass |
| 3rd | NVIDIA RTX 5080 |
| Additional | Ray-Ban Meta glasses, GTC exhibit passes |

## 13.3 Judges

- **Jana van Greunen** -- Director of PyTorch Engineering, Meta
- **Luis Ceze** -- VP of AI Systems Software, NVIDIA
- **Ujval Kapasi** -- VP of AI & HPC Frameworks, NVIDIA
- **Horace He** -- Member of Technical Staff, Thinking Machines Lab

## 13.4 Scoring System

### Points Per Kernel

| Kernel | Correctness | Performance |
|--------|-------------|-------------|
| FP8 Quantization | 100 | 0 (correctness only) |
| Causal Conv1D | 100 | 1000 |
| DeltaNet chunk_fwd_h | 100 | 1000 |
| DeltaNet chunk_fwd_o | 100 | 1000 |
| DeltaNet recompute_w_u | 100 | 1000 |
| **Maximum Total** | **500** | **4000** |

**Grand Total Maximum: 4500 points**

### Scoring Formula

```
Score = CorrectnessPoints + (PerformancePoints * [1 - (rank - 1) / 10])
```

- Correctness: 100 points if ALL test shapes pass
- Performance: Only top 10 performers earn points
  - Rank 1: 100% (1000 pts)
  - Rank 2: 90% (900 pts)
  - Rank 3: 80% (800 pts)
  - ...
  - Rank 10: 10% (100 pts)
  - Rank 11+: 0 performance points

### Performance Metric

- Kernel captured in CUDA graph, replayed with L2 cache clearing
- Graph unrolled to fill ~100ms of GPU time
- Repeated 10 times; runtime = arithmetic mean
- Ranking by **geometric mean** across all benchmark shapes
- Tiebreaker: earlier submission wins

## 13.5 Rules & Requirements

1. **Must use Helion DSL** -- escape hatches (inline_triton, triton_kernel, inline_asm) allowed but <= 30% of LOC
2. **Must pass ALL test input shapes** (rtol/atol within tolerance)
3. **DO NOT autotune on KernelBot** -- hardcode all configs before submission
4. **Single Python file** per kernel submission
5. **Unlimited submissions** -- only best counts
6. **Each submission must include**:
   - Helion kernel implementation
   - One config per test shape
   - One autotuned config per benchmark shape

## 13.6 Submission Workflow

```bash
# Install popcorn-cli
curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash
popcorn register discord
popcorn join <INVITE_CODE>
popcorn setup  # Select "Helion Kernel Challenge"

# Embed directives in submission.py
#!POPCORN leaderboard causal_conv1d
#!POPCORN gpu B200_Nebius

# Test correctness
popcorn submit submission.py --gpu B200_Nebius --leaderboard causal_conv1d --mode test

# Benchmark (non-ranking)
popcorn submit submission.py --gpu B200_Nebius --leaderboard causal_conv1d --mode benchmark

# Official leaderboard submission
popcorn submit submission.py --gpu B200_Nebius --leaderboard causal_conv1d --mode leaderboard

# Profile with Nsight Compute
popcorn submit submission.py --gpu B200_Nebius --leaderboard causal_conv1d --mode profile
```

---

# 14. Kernel-by-Kernel Deep Analysis

## 14.1 Causal Depthwise 1D Convolution (`causal_conv1d`)

### Algorithm
Depthwise convolution with causal (left) padding, used in Mamba/Mamba-2 SSMs:
1. Zero-pad input on the left by (W-1): `x_padded = [zeros, x]`
2. For each channel d independently: `y[b,d,s] = sum(w[d,j] * x_padded[b,d,s+j] for j in range(W)) + bias[d]`
3. Equivalent to `F.conv1d(x_padded, weight.unsqueeze(1), bias, groups=D)`

### Relevance
Causal conv1d is a critical component in Mamba and Mamba-2 architectures for sequence modeling. The "causal" constraint means each output position only depends on current and past input positions.

### Optimization Insights
- Baseline computes w*x 3 times per filter tap and averages
- Optimal: single accumulation loop over W (filter width is small: 3-8)
- W is specialized via `hl.specialize` for loop unrolling
- Use `hl.load` with computed indices for sliding window access
- Parallelize across (B, D, S) dimensions; D and S are the large dimensions
- For B=1 (most benchmark shapes), D is the primary parallelization axis

### Test Shapes
`(B=1,D=64,S=64,W=4)`, `(2,128,128,4)`, `(1,256,256,3)`, `(1,128,64,8)`, `(4,64,128,4)`

### Benchmark Shapes
`(1,768,512,4)`, `(1,768,2048,4)`, `(1,1536,2048,4)`, `(1,2560,2048,4)`, `(1,2560,4096,4)`

---

## 14.3 Gated DeltaNet chunk_fwd_h

### Algorithm
Inter-chunk state recurrence -- the sequential backbone of Gated DeltaNet:

For each chunk c = 0, 1, ..., NT-1:
1. Store state: `h_out[c] = h` (checkpoint for output computation)
2. Compute correction: `v_new[c] = u[c] - w[c] @ h` (subtract state contribution)
3. Compute gating: `alpha = exp(g_end - g_t)` for decay within chunk
4. Update state: `h = h * exp(g_end) + k[c]^T @ (v_new[c] * alpha)`

### Mathematical Background (Gated DeltaNet)
The gated delta rule: `S_t = alpha_t * S_{t-1} + beta_t * (v_t - S_{t-1} * k_t) * k_t^T`
- `alpha_t` controls state decay (gating)
- `beta_t` controls update magnitude
- Uses WY representation for hardware-efficient chunkwise computation
- Within-chunk computation restructured as batched GEMMs

### Optimization Insights
- Baseline computes dot products twice and averages
- Optimal: single `hl.dot` for w@state and k^T@diff
- Sequential across chunks (inherent data dependency), parallel across B*H and V
- K and V dimensions specialized via `hl.specialize` for compile-time optimization
- State tensor is [K, V] per (batch, head) -- fits in registers/shared memory for K,V <= 128
- The inner loop over chunks is the bottleneck -- minimize per-iteration overhead

### Test Shapes
`(B=1,T=64,H=2,K=64,V=64)`, `(2,128,4,64,64)`, `(1,256,4,64,128)`

### Benchmark Shapes
`(1,64,1,64,64)`, `(2,512,3,64,64)`, `(2,1024,3,64,64)`, `(3,1024,4,100,100)`, `(4,1024,4,128,128)`, `(2,1536,4,128,128)`, `(4,2048,8,64,64)`

---

## 14.4 Gated DeltaNet chunk_fwd_o

### Algorithm
Output computation combining local intra-chunk attention with global inter-chunk state:

1. Scale queries: `q_scaled = q * exp(g)`
2. Scale keys: `k_scaled = k * exp(-g)`
3. Local attention: `sim = q_scaled @ k_scaled^T` (with causal mask)
4. Local output: `local = sim @ v_new`
5. Global output: `global = q_scaled @ h[chunk]`
6. Combined: `out = (local + global) * scale` where `scale = K^{-0.5}`

### Optimization Insights
- Baseline computes each dot product twice and averages
- Optimal: single `hl.dot` for QK^T, sim@V, and Q@h
- Causal mask via `hl.arange` comparison (upper triangle zeroed)
- Fully parallel across B*H and T (chunk-level)
- The three matrix multiplications dominate -- optimize their block sizes
- For K=V=64: 64x64 matmuls fit well in tensor cores
- For K=V=128: may need to split across warps

### Test & Benchmark Shapes
Same as chunk_fwd_h

---

## 14.5 Gated DeltaNet recompute_w_u

### Algorithm
WY-transform forward pass computing two matrix products per chunk:

1. `u = A @ (v * beta)` -- weighted value accumulation
2. `w = A @ (k * beta * exp(g))` -- weighted key accumulation with gating

Where A is a `[C, C]` lower-triangular matrix (C=64, the chunk size), obtained from solving a triangular system.

### Reference Implementation
```python
# Reshape into chunks [B, NT, C, H, K/V]
A_c = A.reshape(B, NT, C, H, C).permute(0, 1, 3, 2, 4)  # [B, NT, H, C, C]
u_c = A_c @ (v_c * beta_c.unsqueeze(-1))                   # [B, NT, H, C, V]
w_c = A_c @ (k_c * (beta_c * exp(g_c)).unsqueeze(-1))      # [B, NT, H, C, K]
```

### Optimization Insights
- **Biggest optimization opportunity**: baseline uses element-by-element loops (O(C^2) scalar ops per output element) in TWO passes (forward + backward), then averages
- Optimal: use `hl.dot` for A@k_scaled and A@v_scaled directly -- single matmul per output
- This is a pure matmul kernel: C=64 matrix times C-length vectors (broadcast to K or V)
- Parallelize across B*H and T (chunk boundaries)
- `hl.specialize` for C, K, V to enable compile-time block size selection

### Test & Benchmark Shapes
Same as chunk_fwd_h

---

# 15. NVIDIA B200 GPU Target Platform

## 15.1 Hardware Specifications

| Specification | Value |
|---------------|-------|
| **Architecture** | Blackwell (GB202) |
| **Compute Capability** | 10.0 (sm_100) |
| **CUDA Cores** | 16,896-20,480 |
| **Tensor Cores** | 640 (5th gen) |
| **Streaming Multiprocessors** | ~160 SMs |
| **Memory** | 192 GB HBM3e |
| **Memory Bandwidth** | 8,000 GB/s (8 TB/s) |
| **L2 Cache** | 96-126 MB |
| **Shared Memory per SM** | 228 KB total, 227 KB usable |
| **Tensor Memory (TMEM)** | ~256 KB per SM (new, dedicated to tensor cores) |
| **Registers per SM** | 64K 32-bit registers |
| **Max Registers per Thread** | 255 |
| **Max Warps per SM** | 64 concurrent |
| **Max Thread Blocks per SM** | 32 |
| **FP32 Performance** | 90 TFLOPS |
| **FP16/BF16 Performance** | 2,500 TFLOPS |
| **FP8 Performance** | ~5,000 TFLOPS |
| **FP4 Performance** | 10,000 TFLOPS |
| **TDP** | 1000W |
| **NVLink** | 5th Gen, 1.8 TB/s per GPU |
| **Process** | TSMC 4NP, 208B transistors |

## 15.2 B200-Specific Optimization Tips

1. **Tensor cores are ~2-2.5x faster than H100** -- use larger thread-block tiles (128x128 output sizes)
2. **Experiment with fewer, larger thread blocks** (1-2 CTAs per SM)
3. **Shared memory = 228 KB/SM; TMEM = 256 KB/SM** for tensor ops
4. **[128,128] state matrices (32KB)** fit entirely in SRAM -- compute-bound, not memory-bound
5. **Use `dot_precision="ieee"`** for DeltaNet kernels (as in baselines) for numerical accuracy
6. **Use `hl.specialize()`** for dimensions that should be compile-time constants
7. **Use `spawn` mode** for autotuning if `fork` causes hangs: `HELION_AUTOTUNE_PRECOMPILE=spawn`
8. **Try both Triton and TileIR backends** with and without ACF files
9. **L2 cache is massive (96-126 MB)** -- l2_groupings optimization is less critical but still helps
10. **Warp specialization** is available on Blackwell -- let autotuner explore it

---

# 16. Ecosystem Contribution Opportunities

The hackathon awards special prizes for contributions to the Helion ecosystem, judged separately from kernel performance:

## 16.1 Best Overall Contribution

Significant, well-rounded contributions that improve the framework for all users.

## 16.2 Most Innovative Contribution

Novel, creative approaches that push the boundaries of what Helion can do.

## 16.3 Specific Contribution Categories

### Autotuner Improvements

**Current state**: 7 search algorithms, 3 cache implementations, 3 effort levels.

**Opportunities:**
- **Better search algorithms**: Bayesian optimization with Gaussian processes, multi-fidelity methods (start with small inputs, refine on large), population-based training
- **Transfer learning**: Use configs from similar kernels as starting points; cross-shape config interpolation
- **Warm-starting**: Pre-compute good configs for common kernel patterns (matmul, attention, conv) and use as initial population
- **Faster convergence**: Reduce the 10+ minute tuning time while maintaining quality; adaptive effort based on observed variance
- **Multi-objective**: Optimize for both speed and memory simultaneously
- **Distributed autotuning**: Spread search across multiple GPUs
- **Smart caching**: Better cache invalidation; cross-device config adaptation

### Bug Fixes

**Opportunities:**
- Fix edge cases in masking logic
- Fix dtype promotion issues in generated Triton code
- Fix incorrect stride calculations for non-contiguous tensors
- Address compilation timeouts for complex kernels
- Fix accuracy issues with specific op combinations

### Tooling / Infrastructure

**Opportunities:**
- **Profiling integration**: Built-in Nsight Compute integration; automatic bottleneck detection (compute vs memory bound)
- **Visualization**: Tile layout visualizer; memory access pattern heatmaps; autotuner search space explorer
- **Config management**: Per-hardware config databases; config version control; A/B testing framework
- **CI/CD integration**: GitHub Actions for auto-tuning on PR; regression detection
- **IDE support**: LSP server for Helion; syntax highlighting; inline type hints; kernel previewer
- **Benchmarking suite**: Standardized benchmark suite; performance regression tracking; comparison with other frameworks

### Documentation

**Opportunities:**
- **API reference completeness**: Document all Config parameters with examples and valid ranges
- **Pattern cookbook**: Comprehensive examples for common patterns (attention, normalization, quantization, scanning)
- **Migration guide**: From Triton to Helion; from CUDA to Helion
- **Performance guide**: Best practices for each GPU generation (Hopper, Blackwell)
- **Troubleshooting guide**: Common errors, debugging strategies, performance anti-patterns
- **Architecture guide**: Deep dive into compiler pipeline, autotuner internals, code generation
- **Tutorial series**: Step-by-step for beginners through advanced

### Other Novel Contributions

**Opportunities:**
- **New kernel libraries**: Reusable kernel building blocks (attention variants, normalization, quantization)
- **Model integration**: Drop-in Helion kernels for popular models (Llama, Mamba, DeltaNet)
- **Cross-platform testing**: Verify Helion portability across AMD, Intel, and custom accelerators
- **Training integration**: End-to-end training with Helion kernels (forward + backward)
- **Quantization toolkit**: Comprehensive FP8/FP4/INT8 quantization kernel library
- **Graph compiler integration**: Better integration with torch.compile and CUDA graphs
- **Multi-GPU primitives**: Helion kernels with NCCL/NVLink communication
- **Memory optimization**: Automatic memory planning; kernel fusion across operator boundaries

---

# 17. Sources & References

## Official Helion Resources

- **Helion Documentation**: https://helionlang.com/
- **Helion Tutorials**: https://helionlang.com/helion_tutorials.html
- **Helion API Reference**: https://helionlang.com/api/index.html
- **Helion GitHub Repository**: https://github.com/pytorch/helion
- **PyTorch Blog - Helion**: https://pytorch.org/blog/helion/
- **PyTorch Blog - Accelerating Autotuning**: https://pytorch.org/blog/accelerating-autotuning-in-helion/
- **PyTorch Blog - Portable Paged Attention**: https://pytorch.org/blog/portable-paged-attention-in-helion/
- **PyTorch Conference 2025 Helion Talk**: https://www.youtube.com/watch?v=BW-Ht-5IxgM

## Hackathon Resources

- **Hackathon Rules & Scoring**: https://github.com/gpu-mode/popcorn-cli/blob/main/docs/helion-hackathon.md
- **Reference Kernels**: https://github.com/gpu-mode/reference-kernels/tree/main/problems/helion
- **Popcorn CLI**: https://github.com/gpu-mode/popcorn-cli
- **PyTorch Helion Hackathon Event**: https://pytorch.org/event/helion-hackathon/

## Algorithm Papers

- **Gated DeltaNet**: arXiv:2412.06464 (ICLR 2025) -- Yang, Kautz, Hatamizadeh (NVIDIA)
- **DeltaNet Explained Part I**: https://sustcsonglin.github.io/blog/2024/deltanet-1/
- **DeltaNet Explained Part II**: https://sustcsonglin.github.io/blog/2024/deltanet-2/
- **GatedDeltaNet Implementation**: https://github.com/NVlabs/GatedDeltaNet

## Hardware References

- **NVIDIA Blackwell Tuning Guide**: https://docs.nvidia.com/cuda/blackwell-tuning-guide/
- **NVIDIA FP8 Introduction**: https://developer.nvidia.com/blog/floating-point-8-an-introduction/
- **Causal-Conv1D (Tri Dao)**: https://github.com/Dao-AILab/causal-conv1d

## GPU Programming Landscape

- **Helion and the Evolving GPU Programming Model**: https://ianbarber.blog/2025/10/22/helion-and-the-evolving-gpu-programming-model/
- **GPU MODE Discord**: #helion channel
- **NVIDIA CUDA Compute on GPU MODE**: https://developer.nvidia.com/blog/topping-the-gpu-mode-kernel-leaderboard-with-nvidia-cuda-compute/

---

*Report compiled: March 14, 2026*
*Framework version: Helion (latest from pytorch/helion main branch)*
*Target hardware: NVIDIA B200 (Blackwell, sm_100)*
*Competition: PyTorch Helion Hackathon, GPU MODE*
