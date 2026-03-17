#!/bin/bash
# Sync local code to remote B200 server and run a command
# Usage: ./remote_run.sh <command>
# Examples:
#   ./remote_run.sh "python3 eval.py benchmark causal_conv1d_py/"
#   ./remote_run.sh "ENABLE_TILE=1 HELION_BACKEND=tileir python3 eval.py both gated_deltanet_chunk_fwd_o_py/"
#   ./remote_run.sh "HELION_AUTOTUNE_EFFORT=full python3 eval.py both gated_deltanet_recompute_w_u_py/"

REMOTE_HOST="ubuntu@46.243.147.105"
REMOTE_PASS="bMvoEtw1B6"
REMOTE_DIR="/home/ubuntu/work"
REMOTE_PYTHON="/home/ubuntu/helion_env/bin/python3"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$1" ]; then
    echo "Usage: $0 <command>"
    echo "  Command runs in $REMOTE_DIR with helion_env python"
    echo "  'python3' in your command is auto-replaced with the venv python"
    exit 1
fi

echo "=== Syncing local -> remote ==="

# Sync all kernel submission files + eval/utils
KERNEL_DIRS="causal_conv1d_py gated_deltanet_chunk_fwd_h_py gated_deltanet_chunk_fwd_o_py gated_deltanet_recompute_w_u_py"

for dir in $KERNEL_DIRS; do
    if [ -f "$LOCAL_DIR/$dir/submission.py" ]; then
        sshpass -p "$REMOTE_PASS" scp -o StrictHostKeyChecking=no "$LOCAL_DIR/$dir/submission.py" "$REMOTE_HOST:$REMOTE_DIR/$dir/submission.py" 2>/dev/null
        echo "  Synced $dir/submission.py"
    fi
done

# Sync eval.py, utils.py, and tuning scripts
for f in eval.py utils.py tune_fp8_v2.py tune_fwd_h_v2.py tune_fwd_h_helion.py autotune_deltanet.py autotune_pershape.py; do
    if [ -f "$LOCAL_DIR/$f" ]; then
        sshpass -p "$REMOTE_PASS" scp -o StrictHostKeyChecking=no "$LOCAL_DIR/$f" "$REMOTE_HOST:$REMOTE_DIR/$f" 2>/dev/null
        echo "  Synced $f"
    fi
done

echo ""
echo "=== Running on remote B200 ==="

# Replace python3 with full venv path
CMD=$(echo "$1" | sed "s|python3 |$REMOTE_PYTHON |g")
echo "CMD: $CMD"
echo "---"

sshpass -p "$REMOTE_PASS" ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 "$REMOTE_HOST" "cd $REMOTE_DIR && $CMD"

EXIT_CODE=$?
echo "---"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
