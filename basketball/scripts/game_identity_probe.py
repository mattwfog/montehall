"""Game-level identity inheritance probe (2026-08-25).

The measured wall after spine v3: per-shot evidence is genuinely ambiguous
(a verified-wrong '23' on the spine body; refuted reads on the right body;
82 relink locks with off-roster numbers). The ratified architecture
(contract §1, 08-01) answers ambiguity with GAME-LEVEL identity: every
read across the whole clip pools into who-is-who once, then shots INHERIT.

This probe builds that composition from artifacts that already exist —
no new inference, no pipeline changes:

  entities   relink lock persons (people_relink_smoke segments:
             [[quark_tid, t0_ms, t1_ms, person_id]] + person_meta)
  evidence   (a) shot-sheet reads    shot_attribution rows, quark-keyed
             (b) contact_reads       ByteTrack-keyed, bridged via IoU
             (c) jersey-OCR          identity/ posteriors, same bridge
  prior      person_meta's converged lock number + team
  constraint closed roster vocabulary (the 6-number set) — off-roster
             reads are dropped, never remapped

Per truth-matched shot event: the body that takes the pick decision
(rank-0 shot_attribution row = spine body when the spine claimed the
event) inherits its person's roster-constrained posterior. Scored against
the same truth rows as the e2e chain, at several abstention bars, next to
the shipped picks — same denominator, same matching.

NOT in this probe (v1, deliberately): person merging / (team,number)
exclusivity across locks. Raw inheritance is measured first; exclusivity
is the next rung only if the posteriors are ambiguous.

Usage (inside cvbench, CPU):
    python scripts/game_identity_probe.py /work/out-harvest/hs_clip_20260824 \
        --relink /work/models/quark-v1/hsclip_relink_quark.json \
        --e2e /work/models/quark-v1/e2e_hsclip_spine_v3.json \
        --roster /work/hs_clip_roster.json \
        --out /work/models/quark-v1/game_identity_probe.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from montehall_cv.pipeline.shot_attribution import (
    _identity_posteriors,
    bt_to_quark_person,
    person_at,
    person_index,
    person_pool,
)
from montehall_cv.store.artifacts import read_stage, stage_complete

CONTACT_READ_CONF_MIN = 0.6   # the repo's standing read-accept bar
BARS = (0.0, 0.4, 0.5, 0.6, 0.7)

# person_index/person_at/bt_to_quark_person/person_pool (and the
# channel weights + SEGMENT_SLACK_MS) moved to their canonical home in
# pipeline/shot_attribution.py when inheritance was wired as the 3rd
# identity channel (2026-08-25) — this probe imports back.


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("game_dir", type=Path)
    ap.add_argument("--relink", type=Path, required=True)
    ap.add_argument("--e2e", type=Path, required=True)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--binding-stage", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    job_dir = args.game_dir
    binding = args.binding_stage or job_dir / "quark_binding"
    for need in (binding, job_dir / "shot_attribution",
                 job_dir / "contact_reads"):
        if not stage_complete(need):
            raise SystemExit(f"missing stage: {need}")

    roster = {str(n) for n in json.loads(args.roster.read_text())}
    relink = json.loads(args.relink.read_text())
    pidx = person_index(relink["segments"])
    person_meta = {int(k): v for k, v in relink["person_meta"].items()}

    # ---- pool evidence per person via the canonical pipeline pooling ----
    tallies: dict[int, Counter] = defaultdict(Counter)
    attr_rows = read_stage(job_dir / "shot_attribution").to_pylist()
    sheet_rows = []
    for r in attr_rows:
        if not r["number"]:
            continue
        pid = person_at(pidx, int(r["track_id"]), int(r["ts_anchor_ms"]))
        if pid is None:
            continue
        sheet_rows.append((pid, r["number"],
                           float(r["read_confidence"] or 0.0)))
        if str(r["number"]) in roster:
            tallies[pid]["sheet"] += 1

    bt_person = bt_to_quark_person(job_dir, binding, pidx)

    contact_rows = []
    for r in read_stage(job_dir / "contact_reads").to_pylist():
        if (not r["number"]
                or float(r["confidence"] or 0.0) < CONTACT_READ_CONF_MIN):
            continue
        pid = bt_person.get(int(r["track_id"]))
        if pid is None:
            continue
        contact_rows.append((pid, r["number"], float(r["confidence"])))
        if str(r["number"]) in roster:
            tallies[pid]["contact"] += 1

    ocr_rows = []
    for bt_id, posterior in _identity_posteriors(job_dir).items():
        pid = bt_person.get(bt_id)
        if pid is None:
            continue
        for num, prob in posterior.items():
            ocr_rows.append((pid, num, float(prob)))
            if str(num) in roster:
                tallies[pid]["ocr"] += 1

    posteriors = person_pool(sheet_rows, contact_rows, ocr_rows, roster)

    # ---- shots inherit ----
    e2e = json.loads(args.e2e.read_text())
    pick_body: dict[int, tuple[int, int]] = {}  # event -> (tid, ts_anchor)
    for r in attr_rows:
        if r["rank"] == 0:
            pick_body[int(r["event_id"])] = (int(r["track_id"]),
                                             int(r["ts_anchor_ms"]))
    shipped: dict[int, str | None] = {}
    for r in attr_rows:
        if r["picked"] and r["number"]:
            shipped[int(r["event_id"])] = str(r["number"])

    per_event = []
    for t in e2e["per_event"]:
        if t.get("jersey") is None or t.get("event_id") is None:
            continue
        eid = int(t["event_id"])
        tid, a_ms = pick_body.get(eid, (None, None))
        pid = person_at(pidx, tid, a_ms) if tid is not None else None
        inh = posteriors.get(pid) if pid is not None else None
        meta = person_meta.get(pid, {}) if pid is not None else {}
        per_event.append({
            "event_id": eid,
            "truth_jersey": str(t["jersey"]),
            "shipped_pick": shipped.get(eid),
            "body_tid": tid,
            "person_id": pid,
            "person_lock_number": meta.get("number"),
            "person_team": meta.get("team"),
            "inherited_number": inh[0] if inh else None,
            "inherited_share": round(inh[1], 4) if inh else None,
            "evidence": dict(tallies.get(pid, {})) if pid is not None else {},
        })

    def score(pick_of) -> dict:
        rows = [(e, pick_of(e)) for e in per_event]
        named = [(e, p) for e, p in rows if p is not None]
        hits = sum(1 for e, p in named if p == e["truth_jersey"])
        return {"named": len(named), "hits": hits,
                "of_truth_rows": len(per_event),
                "accuracy_all": round(hits / max(len(per_event), 1), 4),
                "accuracy_of_named": round(hits / max(len(named), 1), 4)}

    summary = {
        "shipped_picks": score(lambda e: e["shipped_pick"]),
        "lock_number_raw": score(
            lambda e: e["person_lock_number"]
            if e["person_lock_number"] in roster else None),
    }
    for bar in BARS:
        summary[f"inherited@{bar}"] = score(
            lambda e, b=bar: e["inherited_number"]
            if e["inherited_share"] is not None
            and e["inherited_share"] >= b else None)

    out = {
        "game_dir": str(job_dir),
        "roster": sorted(roster),
        "persons_with_evidence": len(posteriors),
        "persons_total": len(person_meta),
        "bt_bridged": len(bt_person),
        "summary": summary,
        "per_event": per_event,
        "person_posteriors": {
            str(pid): {"number": n, "share": round(s, 4),
                       "lock_number": person_meta.get(pid, {}).get("number"),
                       "team": person_meta.get(pid, {}).get("team"),
                       "evidence": dict(tallies.get(pid, {}))}
            for pid, (n, s) in sorted(posteriors.items())
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({"summary": summary,
                      "persons_with_evidence": len(posteriors)}, indent=2))


if __name__ == "__main__":
    main()
