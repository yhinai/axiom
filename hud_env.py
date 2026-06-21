"""HUD wrapper for Axiom optimizer trial events.

The B200 benchmark loop writes verifier-produced JSON rows. This environment
lets a HUD agent submit one such row as the answer so HUD can show the live
optimization trace without re-running the expensive benchmark inside HUD.
"""

from __future__ import annotations

import json
from typing import Any

try:
    from hud import Environment
    from hud.types import EvaluationResult, SubScore
except Exception:  # pragma: no cover - local editing should not require HUD.
    Environment = None
    EvaluationResult = None
    SubScore = None


def _score_event(row: dict[str, Any]) -> dict[str, Any]:
    correct = bool(row.get("correct"))
    accepted = bool(row.get("accepted"))
    reward_raw = float(row.get("reward") or 0.0)
    best_ms = row.get("best_geomean_mean_ms_after")
    candidate_ms = row.get("candidate_geomean_mean_ms")
    latency_score = 0.0
    if isinstance(candidate_ms, (int, float)) and candidate_ms > 0:
        # 0.05 ms is a conservative upper bound for these B200 kernel rows.
        latency_score = max(0.0, min(1.0, 1.0 - float(candidate_ms) / 0.05))
    reward = 0.0 if not correct else max(0.0, min(1.0, reward_raw / 2.0))
    if accepted:
        reward = min(1.0, reward + 0.15)
    return {
        "reward": reward,
        "correct": correct,
        "accepted": accepted,
        "latency_score": latency_score,
        "best_ms": best_ms,
        "candidate_ms": candidate_ms,
    }


def _eval_result(row: dict[str, Any]):
    score = _score_event(row)
    if EvaluationResult is None:
        return {"reward": score["reward"], "info": row | score}
    return EvaluationResult(
        reward=score["reward"],
        done=True,
        content=json.dumps(row, sort_keys=True),
        info={
            "axiom": row,
            "score": score,
            "kernel": row.get("kernel"),
            "trial": row.get("trial"),
            "candidate": row.get("candidate"),
            "correct": score["correct"],
            "accepted": score["accepted"],
            "candidate_geomean_mean_ms": score["candidate_ms"],
            "best_geomean_mean_ms_after": score["best_ms"],
        },
        subscores=[
            SubScore(name="correctness", value=1.0 if score["correct"] else 0.0),
            SubScore(name="accepted", value=1.0 if score["accepted"] else 0.0),
            SubScore(name="latency", value=score["latency_score"]),
            SubScore(name="reward", value=score["reward"]),
        ],
    )


def _make_env():
    if Environment is None:
        return None
    try:
        return Environment(name="axiom")
    except TypeError:  # pragma: no cover
        return Environment(id="axiom")


env = _make_env()

if env is not None:

    @env.template(id="optimizer_trial")
    async def optimizer_trial():
        answer = yield (
            "Submit one Axiom optimizer JSON trial row. The row must include "
            "kernel, trial, candidate, correct, accepted, reward, and latency metrics."
        )
        try:
            row = json.loads(answer or "{}")
            if not isinstance(row, dict):
                raise ValueError("trial payload must be a JSON object")
        except Exception as exc:  # noqa: BLE001
            row = {
                "event": "parse_error",
                "correct": False,
                "accepted": False,
                "reward": 0.0,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        yield _eval_result(row)

    axiom_optimizer_trial = optimizer_trial()
    axiom_optimizer_trial.slug = "axiom_optimizer_trial"
    axiom_optimizer_trial.columns = {"project": "axiom", "stream": "optimizer"}


def main() -> int:
    print(json.dumps({"env_id": "axiom", "hud_available": env is not None, "tasks": ["axiom_optimizer_trial"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
