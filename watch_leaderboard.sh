#!/usr/bin/env bash
# Watch the Helion leaderboards. Remote authentication uses SSH keys or an agent.
set -euo pipefail

REMOTE_HOST="${AXIOM_REMOTE_HOST:-ubuntu@46.243.147.105}"
API="${AXIOM_LEADERBOARD_API:-https://site--bot--dxfjds728w5v.code.run}"
GPU="${AXIOM_GPU:-B200_Nebius}"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=yes)
LEADERBOARDS=(
    causal_conv1d
    gated_deltanet_chunk_fwd_h
    gated_deltanet_chunk_fwd_o
    gated_deltanet_recompute_w_u
)

CLI_ID=$(ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "grep cli_id ~/.popcorn.yaml | awk '{print \\$2}'")

if [ -z "$CLI_ID" ]; then
    echo "Failed to read the remote CLI ID"
    exit 1
fi

fetch_leaderboard() {
    curl --fail --silent --show-error \
        -H "X-Popcorn-Cli-Id: $CLI_ID" \
        "$API/submissions/$1/$GPU"
}

display() {
    python3 -c '
import json
import sys

data = json.load(sys.stdin)
if not data:
    print("  (no entries yet)")
for entry in data[:10]:
    rank = entry["rank"]
    name = entry.get("user_name", "???")[:20]
    score = entry.get("submission_score")
    score_text = f"{score:.6e}" if score else "-"
    submitted = entry.get("submission_time", "")[:19].replace("T", " ")
    print(f"  {rank:<4} {name:<20} {score_text:>14}  {submitted}")
'
}

while true; do
    clear
    printf 'HELION HACKATHON LEADERBOARD · %s\n\n' "$(date '+%H:%M:%S')"
    for leaderboard in "${LEADERBOARDS[@]}"; do
        printf '  ▸ %s\n' "$leaderboard"
        fetch_leaderboard "$leaderboard" | display || echo "  (failed to fetch)"
        echo
    done
    echo "Refreshing in 5s…"
    sleep 5
done

