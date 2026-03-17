#!/bin/bash
# Sync local code and run command in a tmux session on remote B200
# Usage: ./remote_tmux.sh <session_name> <command>
# Examples:
#   ./remote_tmux.sh tileir "ENABLE_TILE=1 HELION_BACKEND=tileir python eval.py both gated_deltanet_chunk_fwd_o_py/"
#
# To check output: ./remote_tmux.sh <session_name> --attach
# To list sessions: ./remote_tmux.sh --list

REMOTE_HOST="ubuntu@46.243.147.105"
REMOTE_PASS="bMvoEtw1B6"
REMOTE_DIR="/home/ubuntu/work"
REMOTE_VENV="/home/ubuntu/helion_env/bin"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

SSH_CMD="sshpass -p '$REMOTE_PASS' ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30"
SCP_CMD="sshpass -p '$REMOTE_PASS' scp -o StrictHostKeyChecking=no"

if [ "$1" = "--list" ]; then
    echo "Active tmux sessions on remote:"
    eval $SSH_CMD "$REMOTE_HOST" "tmux list-sessions 2>/dev/null || echo 'No sessions'"
    exit 0
fi

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <session_name> <command>"
    echo "       $0 <session_name> --attach"
    echo "       $0 --list"
    exit 1
fi

SESSION="$1"
CMD="$2"

if [ "$CMD" = "--attach" ]; then
    eval $SSH_CMD -t "$REMOTE_HOST" "tmux attach -t $SESSION 2>/dev/null || tmux new -s $SESSION"
    exit 0
fi

# Sync files
echo "=== Syncing local -> remote ==="
KERNEL_DIRS="causal_conv1d_py gated_deltanet_chunk_fwd_h_py gated_deltanet_chunk_fwd_o_py gated_deltanet_recompute_w_u_py"
for dir in $KERNEL_DIRS; do
    if [ -f "$LOCAL_DIR/$dir/submission.py" ]; then
        eval $SCP_CMD "$LOCAL_DIR/$dir/submission.py" "$REMOTE_HOST:$REMOTE_DIR/$dir/submission.py" 2>/dev/null
    fi
done
for f in eval.py utils.py; do
    [ -f "$LOCAL_DIR/$f" ] && eval $SCP_CMD "$LOCAL_DIR/$f" "$REMOTE_HOST:$REMOTE_DIR/$f" 2>/dev/null
done
echo "  Done syncing"

# Create/replace tmux session and run command
echo "=== Starting tmux session '$SESSION' ==="
echo "CMD: $CMD"
eval $SSH_CMD "$REMOTE_HOST" "tmux kill-session -t $SESSION 2>/dev/null; tmux new-session -d -s $SESSION 'cd $REMOTE_DIR && export PATH=$REMOTE_VENV:\$PATH && $CMD 2>&1 | tee /tmp/$SESSION.log; echo DONE; sleep 3600'"
echo "Session started. Use '$0 $SESSION --attach' to view output."
echo "Or: $0 --list to see all sessions"
