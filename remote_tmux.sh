#!/usr/bin/env bash
# Sync local kernels and run a command in a remote tmux session.
# Authentication must use SSH keys or an SSH agent.
set -euo pipefail

REMOTE_HOST="${AXIOM_REMOTE_HOST:-ubuntu@46.243.147.105}"
REMOTE_DIR="${AXIOM_REMOTE_DIR:-/home/ubuntu/work}"
REMOTE_VENV="${AXIOM_REMOTE_VENV:-/home/ubuntu/helion_env/bin}"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=yes)

if [ "${1:-}" = "--list" ]; then
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "tmux list-sessions 2>/dev/null || echo 'No sessions'"
    exit 0
fi

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <session_name> <command|--attach>"
    echo "       $0 --list"
    exit 1
fi

session="$1"
command="$2"

if [ "$command" = "--attach" ]; then
    ssh "${SSH_OPTS[@]}" -t "$REMOTE_HOST" "tmux attach -t '$session' 2>/dev/null || tmux new -s '$session'"
    exit 0
fi

echo "=== Syncing local -> remote ==="
for dir in causal_conv1d_py gated_deltanet_chunk_fwd_h_py gated_deltanet_chunk_fwd_o_py gated_deltanet_recompute_w_u_py; do
    if [ -f "$LOCAL_DIR/$dir/submission.py" ]; then
        scp "${SSH_OPTS[@]}" "$LOCAL_DIR/$dir/submission.py" "$REMOTE_HOST:$REMOTE_DIR/$dir/submission.py"
    fi
done
for file in eval.py utils.py; do
    if [ -f "$LOCAL_DIR/$file" ]; then
        scp "${SSH_OPTS[@]}" "$LOCAL_DIR/$file" "$REMOTE_HOST:$REMOTE_DIR/$file"
    fi
done

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
    "tmux kill-session -t '$session' 2>/dev/null || true; tmux new-session -d -s '$session' \"cd '$REMOTE_DIR' && export PATH='$REMOTE_VENV':\\\$PATH && $command 2>&1 | tee '/tmp/$session.log'\""
echo "Started remote tmux session: $session"

