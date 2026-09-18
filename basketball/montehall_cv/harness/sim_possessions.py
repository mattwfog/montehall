"""Possession traces with exact truth, from the state simulator.

The simulator (brain/sim_traces.py) runs the latent game forward, so every
possession has a known ending: a made field goal, a miss the defense rebounds,
or a turnover (an offensive rebound continues the possession, per FIBA). This
module replays a sim game and writes each possession twice:

- the TRUTH: how it ended and, for a make, which observed entity shot;
- the TRACE the harness would have seen, degraded the way the real pipeline's
  stages are: ball-control samples exist only on ticks where the simulator
  emitted a ball observation (its measured ~10-45% visibility by ball mode),
  entity ids are the OBSERVED ids (tracker swaps at crossovers included), a
  shot event is detected with probability SHOT_DETECT_P, and its made flag is
  right with probability MADE_FLAG_P.

SHOT_DETECT_P and MADE_FLAG_P are the fixed-camera figures from the state
estimation paper's component table (attempt recall .82, make/miss accuracy
.875). Everything else comes from the simulator. The result scores an
adjudicator's judgment and calibration under controlled, stated noise. It says
nothing about accuracy on real footage.
"""

from __future__ import annotations

import json
import random

from montehall_cv.brain.sim_traces import _Game
from montehall_cv.brain.tokens import CH_BALL, CH_PBP_ANCHOR
from montehall_cv.harness.traces import _passes

SHOT_DETECT_P = 0.82
MADE_FLAG_P = 0.875
MIN_POSSESSION_S = 2.0


class _RecordingGame(_Game):
    """The simulator, also noting per tick what a ball-control stage could see."""

    def __init__(self, key: str, rng: random.Random, duration_s: float):
        super().__init__(key, rng, duration_s)
        self.ticks: list[dict] = []

    def _emit(self, occluded: set[int]) -> None:
        first_new = len(self.tokens)
        super()._emit(occluded)
        new = self.tokens[first_new:]
        holder = self.holder
        self.ticks.append(
            {
                "t_ms": self.t_ms,
                "period": self.period,
                "possession": self.possession,
                "ball_seen": any(t["channel"] == CH_BALL for t in new),
                "holder_obs": self.obs_map[holder] if holder is not None else None,
                "holder_team": (0 if holder < 5 else 1) if holder is not None else None,
                "holder_xy": tuple(self.pos[holder]) if holder is not None else None,
                "obs_map": list(self.obs_map),
            }
        )


def _shots(game: _RecordingGame) -> list[dict]:
    """Resolved shots, in time order: when, by whom (true entity), made or not."""
    shots = []
    for token in game.tokens:
        if token["channel"] != CH_PBP_ANCHOR:
            continue
        payload = json.loads(token["payload_json"])
        shots.append(
            {
                "t_ms": token["t_ms"],
                "made": bool(payload["scoring_play"]),
                "shooter": int(payload["athlete_ids"][0]),
            }
        )
    return sorted(shots, key=lambda s: s["t_ms"])


def possessions_from_game(
    key: str,
    seed: int,
    duration_s: float,
    shot_detect_p: float = SHOT_DETECT_P,
    made_flag_p: float = MADE_FLAG_P,
) -> list[tuple[dict, dict]]:
    """[(trace, truth)] for every possession of one sim game that ended in play.
    The game, and so the truth, depends only on the seed; the two sensor rates
    change only what the trace shows of it."""
    game = _RecordingGame(key, random.Random(seed), duration_s)
    game.run()
    noise = random.Random(seed ^ 0x5EED)  # observation noise, independent of the game
    shots = _shots(game)
    shot_at = {s["t_ms"]: s for s in shots}

    out: list[tuple[dict, dict]] = []
    start = 0
    for i in range(1, len(game.ticks)):
        prev, cur = game.ticks[i - 1], game.ticks[i]
        if cur["period"] != prev["period"]:
            start = i  # a possession cut off by the horn has no ending to judge
            continue
        if cur["possession"] == prev["possession"]:
            continue
        # The flip lands on tick i. A shot resolved on that tick caused it;
        # otherwise the ball was stolen or a pass was picked off.
        cause = shot_at.get(cur["t_ms"])
        if cause is None:
            outcome, scorer = "turnover", None
        elif cause["made"]:
            outcome = "made_fg"
            scorer = cur["obs_map"][cause["shooter"]]
        else:
            outcome, scorer = "missed_fg_dreb", None
        segment = game.ticks[start:i]
        start = i
        if not segment:
            continue
        t0, t1 = segment[0]["t_ms"], cur["t_ms"]
        if (t1 - t0) / 1000 < MIN_POSSESSION_S:
            continue
        offense = prev["possession"]
        trace = _observed_trace(
            len(out), offense, segment, t0, t1, shots, noise, shot_detect_p, made_flag_p
        )
        out.append((trace, {"outcome": outcome, "scorer_entity": scorer}))
    return out


def _observed_trace(
    pid, offense, segment, t0, t1, shots, noise, shot_detect_p, made_flag_p
) -> dict:
    controls = [
        {
            "ts_ms": tick["t_ms"],
            "entity_id": tick["holder_obs"],
            "team_cluster": tick["holder_team"],
            "court_x": tick["holder_xy"][0] + noise.gauss(0, 0.7),
            "court_y": tick["holder_xy"][1] + noise.gauss(0, 0.7),
        }
        for tick in segment
        if tick["ball_seen"] and tick["holder_obs"] is not None
    ]
    shot_events = []
    for shot in shots:
        # t0 is the tick the PREVIOUS possession ended on; a shot resolved then
        # belongs to that possession, so the window is open at t0.
        if not (t0 < shot["t_ms"] <= t1) or noise.random() > shot_detect_p:
            continue
        right = noise.random() < made_flag_p
        shot_events.append(
            {
                "ts_s": round(shot["t_ms"] / 1000, 1),
                "attempt": True,
                "made": shot["made"] if right else not shot["made"],
                "confidence": round(noise.uniform(0.6, 0.95) if right else noise.uniform(0.4, 0.8), 2),
            }
        )
    return {
        "possession_id": pid,
        "start_s": round(t0 / 1000, 1),
        "end_s": round(t1 / 1000, 1),
        "duration_s": round((t1 - t0) / 1000, 1),
        "offense_team_cluster": offense,
        "ball_controls": [
            {
                "ts_s": round(c["ts_ms"] / 1000, 1),
                "entity": c["entity_id"],
                "team": c["team_cluster"],
                "court_x": round(c["court_x"], 1),
                "court_y": round(c["court_y"], 1),
            }
            for c in controls
        ],
        "shot_events": shot_events,
        "passes": _passes(controls),
        "entities_involved": [
            {"entity": eid, "jersey": None} for eid in sorted({c["entity_id"] for c in controls})
        ],
    }


def sample(
    n: int,
    seed: int,
    game_s: float = 1200.0,
    shot_detect_p: float = SHOT_DETECT_P,
    made_flag_p: float = MADE_FLAG_P,
) -> list[tuple[dict, dict]]:
    """n possessions drawn from as many seeded sim games as it takes. Possession
    ids are renumbered to be unique across games."""
    out: list[tuple[dict, dict]] = []
    game_idx = 0
    while len(out) < n:
        games = possessions_from_game(
            f"sim_{seed}_{game_idx}", seed + game_idx, game_s, shot_detect_p, made_flag_p
        )
        for trace, truth in games:
            trace["possession_id"] = len(out)
            out.append((trace, truth))
            if len(out) == n:
                break
        game_idx += 1
    return out
