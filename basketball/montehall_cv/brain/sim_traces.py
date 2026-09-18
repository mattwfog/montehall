"""Headless sim state-trace generator (plan §4 target 1, contract §2).

The latent-state model run FORWARD: a seeded possession/ball-mode state
machine over 10 entities on a real court emits three stages per sim game —

  sim_states/     dense per-tick latent truth (SIM_STATES_SCHEMA)
  sim_positions/  dense per-tick true entity+ball positions (POSITIONS_SCHEMA)
  sim_identity/   the truth permutation, event-sourced (SIM_IDENTITY_SCHEMA):
                  full true->observed mapping at tick 0 + one row per entity
                  per crossover swap — replay in tick order to reconstruct
                  the permutation at any tick (slot-estimator supervision)
  tokens/         the DEGRADED observation stream (TOKENS_SCHEMA), with
                  channel sparsity calibrated to measured real densities
                  (ball visible ~13% blended; board deltas lag 1-5s;
                  clock reads ~2s stride) and the crossover kernel built
                  in: entities within 3 ft occlude, and on separation the
                  OBSERVED id mapping swaps with p=0.3 — the tracker
                  identity-swap the brain must learn to resolve.

v1 simplifications (named, not silent): no free throws, no substitutions,
possession kept through whistles, choreography is drift-toward-targets
rather than plays. Supervision target 1 (mode/possession transition
dynamics + belief through occlusion) does not depend on them.

CLI:
    python -m montehall_cv.brain.sim_traces --out-root /work/sim \
        --games 50 --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from montehall_cv.brain.tokens import (
    CH_BALL,
    CH_CLOCK,
    CH_JERSEY_READ,
    CH_PBP_ANCHOR,
    CH_PLAYER,
    CH_SCORE_DELTA,
    make_token,
    sort_key,
    write_meta,
)
from montehall_cv.store.artifacts import ArtifactWriter, stage_complete
from montehall_cv.store.schemas import (
    POSITIONS_SCHEMA,
    SIM_IDENTITY_SCHEMA,
    SIM_STATES_SCHEMA,
    TOKENS_SCHEMA,
)

TICK_MS = 200
PERIOD_S = 1200.0
N_PERIODS = 2
COURT_X, COURT_Y = 94.0, 50.0
BASKETS = ((5.25, 25.0), (88.75, 25.0))
THREE_FT = 22.146
SHOT_CLOCK_S = 30.0
OCCLUDE_FT = 3.0
SWAP_P = 0.30
MODES = ("held", "dribbled", "ballistic", "dead")
# Observation densities (measured anchors where they exist).
BALL_OBS_P = {"held": 0.10, "dribbled": 0.14, "ballistic": 0.45, "dead": 0.05}
PLAYER_DROP_P = 0.12
CLOCK_STRIDE_S = 2.0
JERSEY_READS_PER_MIN = 0.5

LEGAL_NEXT = {
    "dead": {"dead", "held"},
    "held": {"held", "dribbled", "ballistic", "dead"},
    "dribbled": {"dribbled", "held", "dead"},
    "ballistic": {"ballistic", "held", "dead"},
}


class _Game:
    def __init__(self, key: str, rng: random.Random, duration_s: float):
        self.key = key
        self.rng = rng
        self.period_s = min(PERIOD_S, duration_s / N_PERIODS)
        self.pos = [
            [rng.uniform(10, COURT_X - 10), rng.uniform(8, COURT_Y - 8)]
            for _ in range(10)
        ]
        self.vel = [[0.0, 0.0] for _ in range(10)]
        self.targets = [self._spot(i) for i in range(10)]
        self.retarget_at = [0.0] * 10
        self.jerseys = self._jerseys()
        self.obs_map = list(range(10))  # true entity -> observed id
        self.occluded_since: dict[tuple[int, int], int] = {}
        self.period = 1
        self.clock_s = self.period_s
        self.t_ms = 0
        self.tick = 0
        self.mode = "dead"
        self.holder: int | None = None
        self.possession = rng.randint(0, 1)
        self.shot_clock = SHOT_CLOCK_S
        self.dead_until_s = 3.0
        self.flight: dict | None = None
        self.score = [0, 0]
        self.states: list[dict] = []
        self.positions: list[dict] = []
        self.tokens: list[dict] = []
        self.pending_deltas: list[tuple[float, int, int]] = []
        self.next_clock_read_s = 0.0
        self.makes = 0
        self.identity_rows: list[dict] = []
        self._record_identity(range(10))  # initial mapping (identity)

    # -- setup ---------------------------------------------------------
    def _jerseys(self) -> list[str]:
        nums: list[str] = []
        for _cluster in range(2):
            picks = self.rng.sample(range(0, 56), 5)
            nums.extend(str(n) for n in picks)
        return nums

    def _basket(self, cluster: int) -> tuple[float, float]:
        # cluster 0 attacks the right basket in period 1, sides swap after
        right = (cluster == 0) == (self.period == 1)
        return BASKETS[1] if right else BASKETS[0]

    def _spot(self, entity: int) -> tuple[float, float]:
        half = self.rng.uniform(0, COURT_X / 2)
        x = half if entity < 5 else COURT_X - half
        return (x, self.rng.uniform(5, COURT_Y - 5))

    # -- per-tick dynamics --------------------------------------------
    def _move(self, dt: float) -> None:
        t_s = self.t_ms / 1000
        for i in range(10):
            if t_s >= self.retarget_at[i]:
                cluster = 0 if i < 5 else 1
                bx, by = self._basket(cluster)
                if cluster == self.possession:
                    self.targets[i] = (
                        min(max(bx + self.rng.uniform(-18, 18), 2), COURT_X - 2),
                        min(max(by + self.rng.uniform(-16, 16), 2), COURT_Y - 2),
                    )
                else:
                    self.targets[i] = self._spot(i)
                self.retarget_at[i] = t_s + self.rng.uniform(4, 8)
            tx, ty = self.targets[i]
            for axis, target in ((0, tx), (1, ty)):
                accel = (target - self.pos[i][axis]) * 0.9
                self.vel[i][axis] = max(
                    -24.0,
                    min(24.0, self.vel[i][axis] * 0.8
                        + accel * dt + self.rng.gauss(0, 0.6)),
                )
                self.pos[i][axis] += self.vel[i][axis] * dt
            self.pos[i][0] = min(max(self.pos[i][0], 0.5), COURT_X - 0.5)
            self.pos[i][1] = min(max(self.pos[i][1], 0.5), COURT_Y - 0.5)

    def _ball_xy(self) -> tuple[float, float]:
        if self.flight is not None:
            f = self.flight
            frac = min(1.0, (self.t_ms - f["t0"]) / max(1, f["t1"] - f["t0"]))
            return (
                f["from"][0] + (f["to"][0] - f["from"][0]) * frac,
                f["from"][1] + (f["to"][1] - f["from"][1]) * frac,
            )
        if self.holder is not None:
            hx, hy = self.pos[self.holder]
            return (hx + 0.8, hy)
        bx, by = self._basket(self.possession)
        return (bx, by)

    def _offense(self) -> list[int]:
        return list(range(0, 5)) if self.possession == 0 else list(range(5, 10))

    def _defense(self) -> list[int]:
        return list(range(5, 10)) if self.possession == 0 else list(range(0, 5))

    def _launch(self, kind: str, target_xy, duration_s: float, target_entity=None):
        self.flight = {
            "kind": kind, "from": self._ball_xy(), "to": target_xy,
            "t0": self.t_ms, "t1": self.t_ms + round(duration_s * 1000),
            "target_entity": target_entity, "shooter": self.holder,
        }
        self.holder = None
        self.mode = "ballistic"

    def _shoot(self) -> None:
        assert self.holder is not None
        bx, by = self._basket(self.possession)
        sx, sy = self.pos[self.holder]
        dist = math.hypot(bx - sx, by - sy)
        self._launch("shot", (bx, by), self.rng.uniform(0.9, 1.4))
        assert self.flight is not None
        self.flight["dist_ft"] = dist

    def _resolve_flight(self) -> None:
        f = self.flight
        assert f is not None
        self.flight = None
        if f["kind"] == "pass":
            if self.rng.random() < 0.94:
                self.holder, self.mode = f["target_entity"], "held"
            else:  # picked off
                self.possession ^= 1
                self.holder = self.rng.choice(self._offense())
                self.mode = "held"
                self.shot_clock = SHOT_CLOCK_S
            return
        # shot
        shooter, dist = f["shooter"], f["dist_ft"]
        p_make = 0.55 if dist < 10 else (0.42 if dist < THREE_FT else 0.34)
        made = self.rng.random() < p_make
        pts = 3 if dist >= THREE_FT else 2
        self._emit_shot_anchor(shooter, made, pts, dist)
        if made:
            self.makes += 1
            self.score[self.possession] += pts
            lag = self.rng.uniform(1.0, 5.0)
            self.pending_deltas.append(
                (self.t_ms / 1000 + lag, self.possession, pts))
            self.possession ^= 1
            self.mode = "dead"
            self.dead_until_s = self.rng.uniform(2, 6)
            self.shot_clock = SHOT_CLOCK_S
        else:
            rebounder = self.rng.choice(
                self._offense() if self.rng.random() < 0.28 else self._defense())
            if (rebounder in self._defense()):
                self.possession ^= 1
            self.holder, self.mode = rebounder, "held"
            self.shot_clock = SHOT_CLOCK_S

    def _step_mode(self, dt: float) -> None:
        r = self.rng.random
        if self.mode == "dead":
            self.dead_until_s -= dt
            if self.dead_until_s <= 0:
                self.holder = self.rng.choice(self._offense())
                self.mode = "held"
                self.shot_clock = SHOT_CLOCK_S
            return
        self.clock_s = max(0.0, self.clock_s - dt)
        self.shot_clock -= dt
        if self.mode == "ballistic":
            assert self.flight is not None
            if self.t_ms >= self.flight["t1"]:
                self._resolve_flight()
            return
        if r() < 0.008 * dt:  # whistle (~0.008/s of live play)
            self.mode = "dead"
            self.holder = None
            self.dead_until_s = self.rng.uniform(3, 12)
            return
        if self.mode == "held":
            if self.shot_clock < 5 or r() < 0.06 * dt:
                self._shoot()
            elif r() < 0.30 * dt:
                mates = [e for e in self._offense() if e != self.holder]
                target = self.rng.choice(mates)
                self._launch("pass", tuple(self.pos[target]),
                             self.rng.uniform(0.4, 1.0), target_entity=target)
            elif r() < 0.25 * dt:
                self.mode = "dribbled"
            elif r() < 0.02 * dt:  # stripped
                self.possession ^= 1
                self.holder = self.rng.choice(self._offense())
                self.shot_clock = SHOT_CLOCK_S
        elif self.mode == "dribbled" and r() < 0.35 * dt:
            self.mode = "held"

    def _record_identity(self, entities) -> None:
        for e in entities:
            self.identity_rows.append({
                "game_key": self.key, "tick": self.tick, "t_ms": self.t_ms,
                "true_entity": e, "observed_id": self.obs_map[e],
            })

    # -- occlusion / crossover kernel ---------------------------------
    def _occluded(self) -> set[int]:
        out: set[int] = set()
        pairs_now: set[tuple[int, int]] = set()
        for i in range(10):
            for j in range(i + 1, 10):
                d = math.hypot(self.pos[i][0] - self.pos[j][0],
                               self.pos[i][1] - self.pos[j][1])
                if d < OCCLUDE_FT:
                    out |= {i, j}
                    pairs_now.add((i, j))
                    self.occluded_since.setdefault((i, j), self.tick)
        for pair, since in list(self.occluded_since.items()):
            if pair not in pairs_now:
                if self.tick - since >= 2 and self.rng.random() < SWAP_P:
                    a, b = pair
                    self.obs_map[a], self.obs_map[b] = (
                        self.obs_map[b], self.obs_map[a])
                    self._record_identity((a, b))
                del self.occluded_since[pair]
        return out

    # -- emission ------------------------------------------------------
    def _emit_shot_anchor(self, shooter, made, pts, dist) -> None:
        self.tokens.append(make_token(
            self.key, self.t_ms, CH_PBP_ANCHOR, text="JumpShot",
            value=float(pts if made else 0), period=self.period,
            payload_json=json.dumps({
                "play_id": f"simplay_{len(self.tokens)}",
                "text": "sim jump shot",
                "shooting_play": True, "scoring_play": made,
                "athlete_ids": [str(shooter)],
                "shooter_jersey": self.jerseys[shooter],
                "coord_x": None, "coord_y": None,
                "align_gap_s": 0.0, "dist_ft": round(dist, 1),
            })))

    def _emit(self, occluded: set[int]) -> None:
        bx, by = self._ball_xy()
        self.states.append({
            "game_key": self.key, "tick": self.tick, "t_ms": self.t_ms,
            "period": self.period, "clock_s": round(self.clock_s, 2),
            "ball_mode": self.mode, "ball_x": round(bx, 2),
            "ball_y": round(by, 2), "holder_entity": self.holder,
            "possession_cluster": self.possession,
            "score_home": self.score[0], "score_guest": self.score[1],
            "dead_ball": self.mode == "dead",
        })
        for i in range(10):
            self.positions.append({
                "job_id": self.key, "frame_idx": self.tick, "ts_ms": self.t_ms,
                "cls": 0, "entity_id": i, "team_cluster": 0 if i < 5 else 1,
                "court_x": round(self.pos[i][0], 2),
                "court_y": round(self.pos[i][1], 2), "court_conf": 1.0,
            })
            if i not in occluded and self.rng.random() > PLAYER_DROP_P:
                self.tokens.append(make_token(
                    self.key, self.t_ms, CH_PLAYER,
                    entity_id=self.obs_map[i],
                    team_cluster=0 if i < 5 else 1,
                    court_x=round(self.pos[i][0] + self.rng.gauss(0, 0.7), 2),
                    court_y=round(self.pos[i][1] + self.rng.gauss(0, 0.7), 2),
                    conf=round(self.rng.uniform(0.6, 0.99), 2)))
            if (i not in occluded
                    and self.rng.random() < JERSEY_READS_PER_MIN * TICK_MS / 60000):
                self.tokens.append(make_token(
                    self.key, self.t_ms, CH_JERSEY_READ,
                    text=self.jerseys[i],
                    conf=round(self.rng.uniform(0.85, 0.99), 2),
                    payload_json=json.dumps(
                        {"track_id": self.obs_map[i], "stage": "sim"})))
        self.positions.append({
            "job_id": self.key, "frame_idx": self.tick, "ts_ms": self.t_ms,
            "cls": 1, "entity_id": None, "team_cluster": None,
            "court_x": round(bx, 2), "court_y": round(by, 2),
            "court_conf": 1.0,
        })
        if self.rng.random() < BALL_OBS_P[self.mode]:
            self.tokens.append(make_token(
                self.key, self.t_ms, CH_BALL,
                court_x=round(bx + self.rng.gauss(0, 0.5), 2),
                court_y=round(by + self.rng.gauss(0, 0.5), 2),
                conf=round(self.rng.uniform(0.5, 0.95), 2)))
        t_s = self.t_ms / 1000
        if t_s >= self.next_clock_read_s:
            self.tokens.append(make_token(
                self.key, self.t_ms, CH_CLOCK, value=round(self.clock_s, 1),
                period=self.period,
                conf=round(self.rng.uniform(0.88, 1.0), 2)))
            self.next_clock_read_s = t_s + CLOCK_STRIDE_S + self.rng.uniform(-0.3, 0.3)
        for due_s, cluster, pts in list(self.pending_deltas):
            if t_s >= due_s:
                self.tokens.append(make_token(
                    self.key, self.t_ms, CH_SCORE_DELTA,
                    text="home" if cluster == 0 else "guest",
                    value=float(pts), conf=0.95,
                    payload_json=json.dumps({"before": None, "after": None})))
                self.pending_deltas.remove((due_s, cluster, pts))

    # -- main loop -----------------------------------------------------
    def run(self) -> None:
        dt = TICK_MS / 1000
        while self.period <= N_PERIODS:
            prev_mode = self.mode
            self._move(dt)
            self._step_mode(dt)
            assert self.mode in LEGAL_NEXT[prev_mode], (prev_mode, self.mode)
            occluded = self._occluded()
            self._emit(occluded)
            self.tick += 1
            self.t_ms += TICK_MS
            if self.clock_s <= 0:
                self.period += 1
                if self.period <= N_PERIODS:
                    self.clock_s = self.period_s
                    self.mode = "dead"
                    self.holder = None
                    self.dead_until_s = self.rng.uniform(3, 8)


def generate_game(out_root: Path, key: str, seed: int,
                  duration_s: float = N_PERIODS * PERIOD_S) -> dict:
    game_dir = Path(out_root) / key
    if stage_complete(game_dir / "tokens"):
        return {"game_key": key, "skipped": True}
    game = _Game(key, random.Random(seed), duration_s)
    game.run()

    for stage, schema, rows in (
        ("sim_states", SIM_STATES_SCHEMA, game.states),
        ("sim_positions", POSITIONS_SCHEMA, game.positions),
        ("sim_identity", SIM_IDENTITY_SCHEMA, game.identity_rows),
    ):
        writer = ArtifactWriter(game_dir / stage, schema)
        writer.add_many(rows)
        writer.close()

    game.tokens.sort(key=sort_key)
    counts: dict[str, int] = {}
    writer = ArtifactWriter(game_dir / "tokens", TOKENS_SCHEMA)
    for idx, row in enumerate(game.tokens):
        row["token_idx"] = idx
        counts[row["channel"]] = counts.get(row["channel"], 0) + 1
        writer.add(row)
    writer.close()
    meta = write_meta(game_dir / "tokens", key, counts, ["sim"])
    meta.update({"ticks": game.tick, "makes": game.makes,
                 "final_score": list(game.score), "seed": seed,
                 "identity_swaps": (len(game.identity_rows) - 10) // 2,
                 "jerseys": game.jerseys})
    (game_dir / "sim_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--duration-s", type=float, default=N_PERIODS * PERIOD_S)
    args = ap.parse_args()
    for i in range(args.games):
        key = f"sim_{args.seed:04d}_{i:03d}"
        meta = generate_game(args.out_root, key, args.seed * 100_000 + i,
                             args.duration_s)
        print(f"SIMTRACE: {key} "
              f"{'skipped' if meta.get('skipped') else meta['total']}",
              flush=True)
    print("SIMTRACES_DONE", flush=True)


if __name__ == "__main__":
    main()
