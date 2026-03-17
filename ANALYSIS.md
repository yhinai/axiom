# Helion Hackathon - Implementation Analysis & Optimization Plan

## Scoring System
- **Correctness**: 100 points per kernel (must pass ALL test shapes)
- **Performance**: 0-1000 points based on runtime ranking
- **Formula**: Score = CorrectnessPoints + (PerformancePoints × [1 − (rank − 1) / 10])
- Only top 10 performers per kernel earn performance points

## 4 Kernels Implemented

### 1. Causal Depthwise 1D Convolution (`causal_conv1d_py/`)
**Algorithm**: Depthwise conv1d with causal (left) padding
- Input: `x [B, D, S]`, weight `[D, W]`, bias `[D]`
- Zero-pad left by W-1 → sliding dot product per channel
- Used in Mamba/Mamba-2 state-space models

**Optimization vs Baseline**:
- Baseline computed w*x 3× per position and averaged
- Our version: single accumulation loop over filter width W
- Efficient memory access: coalesced reads along sequence dimension

**Test shapes**: (1,64,64,4), (2,128,128,4), (1,256,256,3), (1,128,64,8), (4,64,128,4)
**Benchmark shapes**: (1,768,512,4), (1,768,2048,4), (1,1536,2048,4), (1,2560,2048,4), (1,2560,4096,4)

### 2. Gated DeltaNet chunk_fwd_h (`gated_deltanet_chunk_fwd_h_py/`)
**Algorithm**: Inter-chunk state recurrence for Gated DeltaNet attention
- Sequential state update across chunks of size C=64
- For each chunk: store h → compute v_new = u - w@h → gate → update h
- Parallelized across B*H and V dimensions

**Optimization vs Baseline**:
- Baseline computed proj/upd 2× each and averaged
- Our version: single dot product for projection and update
- Key: `hl.dot` for w@state and k^T @ diff

**Shapes**: (B, T, H, K, V) with K,V in {64, 100, 128}

### 3. Gated DeltaNet chunk_fwd_o (`gated_deltanet_chunk_fwd_o_py/`)
**Algorithm**: Output computation combining local + global attention
- Local: causal QK^T @ V within each chunk
- Global: Q @ h[chunk] from inter-chunk state
- Combined: (local + global) * scale

**Optimization vs Baseline**:
- Baseline computed sim/local/global 2× each and averaged
- Our version: single dot for QK^T, single dot for sim@V, single dot for Q@h
- Causal mask via `hl.arange` comparison

### 4. Gated DeltaNet recompute_w_u (`gated_deltanet_recompute_w_u_py/`)
**Algorithm**: WY-transform forward: w = A @ (k * beta * exp(g)), u = A @ (v * beta)
- A is chunk-local lower-triangular matrix [C, C]
- Two matrix multiplications per chunk

**Optimization vs Baseline**:
- Baseline iterated element-by-element in TWO loops (forward+backward) and averaged
- Our version: uses `hl.dot` for A @ k_scaled and A @ v_scaled directly
- This is the biggest improvement - O(C²) → O(C) matmul operations

## Performance Tuning Strategy

### Config Parameters to Autotune (on B200 GPU):
1. `block_sizes` - tile dimensions for parallel execution
2. `num_warps` - 1,2,4,8 warps per block
3. `num_stages` - pipeline depth (1-4)
4. `indexing` - 'block_ptr' vs 'pointer'
5. `pid_type` - 'flat' vs 'persistent_blocked'

### Advanced Optimizations (after correctness):
- ACF files from `/opt/booster_pack/` for PTXAS optimization
- TileIR backend (`ENABLE_TILE=1` + `HELION_BACKEND=tileir`)
- `loop_orders`, `l2_groupings`, `range_unroll_factors`

## Submission Workflow
```bash
# Local testing
python eval.py test <problem_dir>/
python eval.py benchmark <problem_dir>/

# Submit to leaderboard
popcorn submit submission.py --gpu B200_Nebius --leaderboard <name> --mode test
popcorn submit submission.py --gpu B200_Nebius --leaderboard <name> --mode benchmark
popcorn submit submission.py --gpu B200_Nebius --leaderboard <name> --mode leaderboard
```

## Key Constraints
- Must use Helion DSL (inline_triton/asm ≤30% LOC)
- Accuracy tolerance: rtol=1e-2, atol=1e-2 for deltanet; rtol=1e-3, atol=1e-3 for fp8
- Do NOT autotune on KernelBot - hardcode all configs
- CUDA graphs with L2 cache clearing for benchmarking
