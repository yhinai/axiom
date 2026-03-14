#!/bin/bash
# Sync local code to remote B200 server and run a command
# Usage: ./remote_run.sh <command>
# Examples:
#   ./remote_run.sh "python eval.py both fp8_quant_py/"
#   ./remote_run.sh "python eval.py benchmark causal_conv1d_py/"
#   ./remote_run.sh "ENABLE_TILE=1 HELION_BACKEND=tileir python eval.py both gated_deltanet_chunk_fwd_o_py/"
#   ./remote_run.sh "HELION_AUTOTUNE_EFFORT=full python eval.py both gated_deltanet_recompute_w_u_py/"

REMOTE_HOST="ubuntu@46.243.147.105"
REMOTE_PASS="bMvoEtw1B6"
REMOTE_DIR="/home/ubuntu/work"
REMOTE_VENV="/home/ubuntu/helion_env/bin"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

SSH_CMD="sshpass -p '$REMOTE_PASS' ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30"
SCP_CMD="sshpass -p '$REMOTE_PASS' scp -o StrictHostKeyChecking=no"

if [ -z "$1" ]; then
    echo "Usage: $0 <command>"
    echo "  Command runs in $REMOTE_DIR with helion_env activated"
    exit 1
fi

echo "=== Syncing local -> remote ==="

# Sync all kernel submission files + eval/utils
KERNEL_DIRS="fp8_quant_py causal_conv1d_py gated_deltanet_chunk_fwd_h_py gated_deltanet_chunk_fwd_o_py gated_deltanet_recompute_w_u_py"

for dir in $KERNEL_DIRS; do
    if [ -f "$LOCAL_DIR/$dir/submission.py" ]; then
        eval $SCP_CMD "$LOCAL_DIR/$dir/submission.py" "$REMOTE_HOST:$REMOTE_DIR/$dir/submission.py" 2>/dev/null
        echo "  Synced $dir/submission.py"
    fi
done

# Sync eval.py and utils.py
for f in eval.py utils.py; do
    if [ -f "$LOCAL_DIR/$f" ]; then
        eval $SCP_CMD "$LOCAL_DIR/$f" "$REMOTE_HOST:$REMOTE_DIR/$f" 2>/dev/null
        echo "  Synced $f"
    fi
done

# Sync any extra .py files in the root (autotune scripts etc)
for f in "$LOCAL_DIR"/*.py; do
    if [ -f "$f" ]; then
        fname=$(basename "$f")
        if [ "$fname" != "eval.py" ] && [ "$fname" != "utils.py" ]; then
            eval $SCP_CMD "$f" "$REMOTE_HOST:$REMOTE_DIR/$fname" 2>/dev/null
            echo "  Synced $fname"
        fi
    fi
done

echo ""
echo "=== Running on remote B200 ==="
echo "CMD: $1"
echo "---"

# Run the command with helion_env activated
eval $SSH_CMD "$REMOTE_HOST" "cd $REMOTE_DIR && export PATH=$REMOTE_VENV:\$PATH && $1"

EXIT_CODE=$?
echo "---"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
