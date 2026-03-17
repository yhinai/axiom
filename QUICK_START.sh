#!/bin/bash
# ============================================
# Helion Hackathon Quick Start Script
# Run this on the B200 GPU machine
# ============================================

echo "=== Helion Hackathon Setup ==="

# Step 1: Register with Discord (already done if popcorn installed)
# popcorn register discord

# Step 2: Join with invite code (replace with actual code)
# popcorn join <YOUR_INVITE_CODE>

# Step 3: Install Helion
pip install helion 2>/dev/null || echo "Helion already installed or install failed"

# Step 4: Install TileIR backend (for Blackwell GPUs)
# pip install https://github.com/triton-lang/Triton-to-tile-IR/releases/download/v3.6.0-rc1/nvtriton-3.6.0-cp312-cp312-linux_x86_64.whl

# ============================================
# LOCAL TESTING
# ============================================

echo ""
echo "=== Local Testing Commands ==="
echo "cd /path/to/helion"
echo ""
echo "# Test correctness (from helion/ root):"
echo "python eval.py test causal_conv1d_py/"
echo "python eval.py test gated_deltanet_chunk_fwd_h_py/"
echo "python eval.py test gated_deltanet_chunk_fwd_o_py/"
echo "python eval.py test gated_deltanet_recompute_w_u_py/"
echo ""
echo "# Benchmark:"
echo "python eval.py benchmark causal_conv1d_py/"
echo ""
echo "# Both test + benchmark:"
echo "python eval.py both causal_conv1d_py/"

# ============================================
# LEADERBOARD SUBMISSIONS
# ============================================

echo ""
echo "=== Submission Commands ==="
echo ""
echo "# Test mode (verify correctness on KernelBot):"
echo "popcorn submit causal_conv1d_py/submission.py --gpu B200_Nebius --leaderboard causal_conv1d --mode test"
echo "popcorn submit gated_deltanet_chunk_fwd_h_py/submission.py --gpu B200_Nebius --leaderboard gated_deltanet_chunk_fwd_h --mode test"
echo "popcorn submit gated_deltanet_chunk_fwd_o_py/submission.py --gpu B200_Nebius --leaderboard gated_deltanet_chunk_fwd_o --mode test"
echo "popcorn submit gated_deltanet_recompute_w_u_py/submission.py --gpu B200_Nebius --leaderboard gated_deltanet_recompute_w_u --mode test"
echo ""
echo "# Leaderboard mode (official scored submission):"
echo "popcorn submit causal_conv1d_py/submission.py --gpu B200_Nebius --leaderboard causal_conv1d --mode leaderboard"
echo "popcorn submit gated_deltanet_chunk_fwd_h_py/submission.py --gpu B200_Nebius --leaderboard gated_deltanet_chunk_fwd_h --mode leaderboard"
echo "popcorn submit gated_deltanet_chunk_fwd_o_py/submission.py --gpu B200_Nebius --leaderboard gated_deltanet_chunk_fwd_o --mode leaderboard"
echo "popcorn submit gated_deltanet_chunk_fwd_o_py/submission.py --gpu B200_Nebius --leaderboard gated_deltanet_chunk_fwd_o --mode leaderboard"
echo "popcorn submit gated_deltanet_recompute_w_u_py/submission.py --gpu B200_Nebius --leaderboard gated_deltanet_recompute_w_u --mode leaderboard"

# ============================================
# TILEIR BACKEND (optional, for Blackwell)
# ============================================

echo ""
echo "=== TileIR Backend (Blackwell only) ==="
echo "export ENABLE_TILE=1"
echo "export HELION_BACKEND=tileir"
echo "export TILEIR_ENABLE_APPROX=1  # optional: fast math for attention"
echo "export TILEIR_ENABLE_FTZ=1     # optional: flush-to-zero"

# ============================================
# AUTOTUNING
# ============================================

echo ""
echo "=== Autotuning ==="
echo "# Get default config (prints to stderr):"
echo "HELION_AUTOTUNE_EFFORT=none python -c 'from submission import custom_kernel; ...'"
echo ""
echo "# Quick autotune:"
echo "HELION_AUTOTUNE_EFFORT=quick python eval.py benchmark causal_conv1d_py/"
echo ""
echo "# Full autotune (takes ~10+ min per shape):"
echo "HELION_AUTOTUNE_EFFORT=full python eval.py benchmark causal_conv1d_py/"
echo ""
echo "# With ACF files:"
echo "ls /opt/booster_pack/*.acf 2>/dev/null || echo 'No ACF files found'"

echo ""
echo "=== Ready! Waiting for invite code... ==="
