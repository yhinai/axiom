#!/usr/bin/env bash
# Watch top 10 for all 5 kernel leaderboards, refreshing every 5 seconds.
# Usage: ./watch_leaderboard.sh

set -euo pipefail

HOST="ubuntu@46.243.147.105"
PASS="bMvoEtw1B6"
API="https://site--bot--dxfjds728w5v.code.run"
GPU="B200_Nebius"
LEADERBOARDS=(
    causal_conv1d
    gated_deltanet_chunk_fwd_h
    gated_deltanet_chunk_fwd_o
    gated_deltanet_recompute_w_u
)

# Get CLI_ID from remote server once
CLI_ID=$(sshpass -p "$PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$HOST" \
    "grep cli_id ~/.popcorn.yaml | awk '{print \$2}'" 2>/dev/null)

if [ -z "$CLI_ID" ]; then
    echo "Failed to get CLI ID from server"
    exit 1
fi

fetch_leaderboard() {
    local lb=$1
    curl -s -H "X-Popcorn-Cli-Id: $CLI_ID" "$API/submissions/$lb/$GPU" 2>/dev/null
}

display() {
    python3 -c "
import sys, json

data = json.load(sys.stdin)
if not data:
    print('  (no entries yet)')
else:
    print(f'  {\"#\":<4} {\"User\":<20} {\"Score\":>14}  {\"Submitted\"}')
    print(f'  {\"─\"*4} {\"─\"*20} {\"─\"*14}  {\"─\"*19}')
    for entry in data[:10]:
        rank = entry['rank']
        name = entry.get('user_name', '???')[:20]
        score = entry.get('submission_score')
        score_s = f'{score:.6e}' if score else '-'
        time = entry.get('submission_time', '')[:19].replace('T', ' ')
        marker = ' <--' if name in sys.argv[1:] else ''
        print(f'  {rank:<4} {name:<20} {score_s:>14}  {time}{marker}')
" "$@"
}

# Optional: pass your username to highlight your entries
MY_USER="${1:-}"

while true; do
    clear
    echo "╔═══════════════════════════════════════════════════════════════════════╗"
    echo "║  HELION HACKATHON LEADERBOARD (Top 10)    $(date '+%H:%M:%S')                    ║"
    echo "╚═══════════════════════════════════════════════════════════════════════╝"
    echo ""

    for lb in "${LEADERBOARDS[@]}"; do
        echo "  ▸ $lb"
        fetch_leaderboard "$lb" | display "$MY_USER" 2>/dev/null || echo "  (failed to fetch)"
        echo ""
    done

    echo "  Refreshing in 5s... (Ctrl+C to stop)"
    sleep 5
done
