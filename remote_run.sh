#!/usr/bin/env bash
# Sync local kernels to the configured B200 host and run a command.
# Authentication must use SSH keys or an SSH agent.
set -euo pipefail

REMOTE_HOST="${AXIOM_REMOTE_HOST:-ubuntu@46.243.147.105}"
REMOTE_DIR="${AXIOM_REMOTE_DIR:-/home/ubuntu/work}"
REMOTE_PYTHON="${AXIOM_REMOTE_PYTHON:-/home/ubuntu/helion_env/bin/python3}"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=yes)

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <command>"
    exit 1
fi

echo "=== Syncing local -> remote ==="
KERNEL_DIRS=(
    causal_conv1d_py
    gated_deltanet_chunk_fwd_h_py
    gated_deltanet_chunk_fwd_o_py
    gated_deltanet_recompute_w_u_py
)

for dir in "${KERNEL_DIRS[@]}"; do
    if [ -f "$LOCAL_DIR/$dir/submission.py" ]; then
        scp "${SSH_OPTS[@]}" "$LOCAL_DIR/$dir/submission.py" "$REMOTE_HOST:$REMOTE_DIR/$dir/submission.py"
        echo "  Synced $dir/submission.py"
    fi
done

for file in eval.py utils.py tune_fwd_h_v2.py tune_fwd_h_helion.py autotune_deltanet.py autotune_pershape.py; do
    if [ -f "$LOCAL_DIR/$file" ]; then
        scp "${SSH_OPTS[@]}" "$LOCAL_DIR/$file" "$REMOTE_HOST:$REMOTE_DIR/$file"
        echo "  Synced $file"
    fi
done

command="${1//python3 /$REMOTE_PYTHON }"
echo "=== Running on remote B200 ==="
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "cd '$REMOTE_DIR' && $command"

