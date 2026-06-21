#!/usr/bin/env bash
set -euo pipefail

# Deploy Axiom to ssh modal and start the real B200 kernel optimizer in tmux.
# Secrets are read from local environment or ~/.hud/.env; they are not stored
# in the repository.

REMOTE_HOST="${REMOTE_HOST:-modal}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/axiom}"
SESSION="${SESSION:-axiom-kernel-improve}"
DURATION_HOURS="${DURATION_HOURS:-5}"
OUT_DIR="${OUT_DIR:-runs/modal-b200-axiom}"
LOG_DIR="${LOG_DIR:-runs/logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_CANDIDATES="${MAX_CANDIDATES:-0}"
KERNEL_WORKERS="${KERNEL_WORKERS:-2}"

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR' '$REMOTE_DIR/$LOG_DIR' ~/.hud"
COPYFILE_DISABLE=1 tar -C "$LOCAL_ROOT" \
  --exclude './.git' \
  --exclude './__pycache__' \
  --exclude './runs' \
  -czf - . | ssh "$REMOTE_HOST" "cd '$REMOTE_DIR' && tar -xzf -"

if [[ -f "$HOME/.hud/.env" ]]; then
  scp -q "$HOME/.hud/.env" "$REMOTE_HOST:~/.hud/.env"
  ssh "$REMOTE_HOST" "chmod 600 ~/.hud/.env"
fi

REMOTE_RUNNER="$REMOTE_DIR/run_${SESSION}.sh"
REMOTE_CMD=$(cat <<EOF
cd $REMOTE_DIR
if [ -d .venv ]; then . .venv/bin/activate; fi
export PYTHONUNBUFFERED=1
mkdir -p $LOG_DIR
$PYTHON_BIN scripts/axiom_optimizer.py \\
  --all-kernels \\
  --duration-hours $DURATION_HOURS \\
  --out-dir $OUT_DIR \\
  --kernel-workers $KERNEL_WORKERS \\
  --loop \\
  ${MAX_CANDIDATES:+--max-candidates $MAX_CANDIDATES} \\
  2>&1 | tee $LOG_DIR/$SESSION.log
EOF
)

printf '%s\n' "$REMOTE_CMD" | ssh "$REMOTE_HOST" "cat > '$REMOTE_RUNNER' && chmod +x '$REMOTE_RUNNER'"
ssh "$REMOTE_HOST" "tmux kill-session -t '$SESSION' 2>/dev/null || true; tmux new-session -d -s '$SESSION' 'bash $REMOTE_RUNNER'"

cat <<EOF
Started Axiom Modal optimizer.
  host:     $REMOTE_HOST
  session:  $SESSION
  repo:     $REMOTE_DIR
  out:      $REMOTE_DIR/$OUT_DIR
  log:      $REMOTE_DIR/$LOG_DIR/$SESSION.log

Watch:
  ssh $REMOTE_HOST 'tmux attach -t $SESSION'
  ssh $REMOTE_HOST 'tail -f $REMOTE_DIR/$LOG_DIR/$SESSION.log'
EOF
