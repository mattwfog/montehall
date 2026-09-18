"""ESPN public PBP pairing — the label source for broadcast games.

Two pairing paths, one output contract (`<stem>.pbp.json` = full ESPN
summary: plays with clock/shootingPlay/scoreValue/participants/coordinate
+ boxscore jersey map; resumable: existing file = skip; fetch-loop rules):

1. Network scoreboard (legacy, ACC-scoped): scoreboard by date lists games,
   summary fetched per match.
2. Local corpus (Ask 0a, 2026-07-16): match titles against the season-pull
   corpus (`pbp-corpus/<league>/<season>/<event_id>.json`, full D1) — zero
   network, full coverage, and a measured pairing report. Title season
   suffix ("2025-26 ..." = regular season, "2026 ... Tournament" = March)
   plus any explicit title date disambiguate home-and-home rematches;
   still-ambiguous titles are reported and NOT paired (abstain > wrong).

CLI:
    python -m montehall_cv.eval.espn_pbp --videos-dir DIR --out-dir DIR \
        --dates 20260308-20260315            # network path
    python -m montehall_cv.eval.espn_pbp --videos-dir DIR --out-dir DIR \
        --pbp-corpus ~/cv-bench/pbp-corpus   # local-corpus path
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
UA = {"User-Agent": "Mozilla/5.0 (montehall research)"}
ACC_GROUP = "2"

# video-title token -> ESPN team `location`; identity unless listed.
TITLE_TO_ESPN = {
    "Cal": "California",
    "Pitt": "Pittsburgh",
}


def _get(url: str, retries: int = 3) -> dict:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as err:  # noqa: BLE001 — transient net errors retry
            last_err = err
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"ESPN fetch failed after {retries} tries: {url}") from last_err


_SUFFIX_RE = re.compile(r"\s+Full\s+(?:Game|Match)|\s+Game\s+Replay", re.IGNORECASE)
_VS_RE = re.compile(r"\s+vs\.?\s+", re.IGNORECASE)


def matchup_from_title(filename: str) -> tuple[str, str] | None:
    """'A vs. B Full Game Replay | ...' -> (title_token_a, title_token_b).

    Handles the channel's title variants: 'vs' with or without the period,
    'Full Game Replay' / 'Full Match Replay' / bare 'Game Replay'. Tokens
    pass through TITLE_TO_ESPN; alt-name resolution against the corpus index
    happens in pair_videos_local.
    """
    stem = _SUFFIX_RE.split(Path(filename).stem)[0]
    parts = _VS_RE.split(stem, maxsplit=1)
    if len(parts) != 2:
        return None
    a, b = parts
    return (
        TITLE_TO_ESPN.get(a.strip(), a.strip()),
        TITLE_TO_ESPN.get(b.strip(), b.strip()),
    )


def scoreboard_events(dates: list[str]) -> dict[frozenset[str], dict]:
    """{frozenset({loc_a, loc_b}): {'event_id', 'date', 'name'}} over dates."""
    out: dict[frozenset[str], dict] = {}
    for d in dates:
        data = _get(f"{BASE}/scoreboard?dates={d}&groups={ACC_GROUP}&limit=300")
        for event in data.get("events", []):
            comps = (event.get("competitions") or [{}])[0].get("competitors") or []
            locs = frozenset(
                str((c.get("team") or {}).get("location", "")) for c in comps
            )
            if len(locs) == 2:
                out[locs] = {
                    "event_id": str(event["id"]),
                    "date": d,
                    "name": event.get("name", ""),
                }
    return out


def fetch_summary(event_id: str) -> dict:
    return _get(f"{BASE}/summary?event={event_id}")


def pbp_path_for(video: Path, out_dir: Path) -> Path:
    return out_dir / f"{video.stem}.pbp.json"


# --- local-corpus pairing (Ask 0a) -------------------------------------------

_TITLE_DATE_RES = (
    # "⧸" (U+29F8) is yt-dlp's sanitized "/" in filenames
    re.compile(r"(?P<m>\d{1,2})[/.⧸-](?P<d>\d{1,2})[/.⧸-](?P<y>\d{2,4})"),
    re.compile(
        r"(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+"
        r"(?P<d>\d{1,2}),?\s*(?P<y>\d{4})?",
        re.IGNORECASE,
    ),
)
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
)}
_SEASON_RE = re.compile(r"(?P<y1>20\d{2})-(?P<y2>\d{2})")
_TOURNAMENT_RE = re.compile(r"\b(?P<y>20\d{2})\b(?!-).*\bTour", re.IGNORECASE)
_BARE_YEAR_RE = re.compile(r"[｜|]\s*(?P<y>20\d{2})\s")


def title_date_window(filename: str) -> tuple[str, str] | None:
    """Title suffix -> (YYYYMMDD lo, YYYYMMDD hi) inclusive, or None.

    Explicit date beats season inference. "2025-26 ..." = regular season
    window (Nov 1 - Apr 30); "2026 ... Tour[nament]" = March window.
    Parses the raw basename (not Path.stem — a sanitized-slash date like
    1⧸16⧸2026 must survive, and a literal "/" must not truncate).
    """
    stem = filename.rsplit(".", 1)[0]
    for pat in _TITLE_DATE_RES:
        m = pat.search(stem)
        if not m:
            continue
        g = m.groupdict()
        month = _MONTHS[g["mon"][:3].lower()] if "mon" in g and g.get("mon") else int(g["m"])
        year_s = g.get("y")
        if not year_s:
            continue  # month/day with no year: not trustworthy alone
        year = int(year_s)
        if year < 100:
            year += 2000
        day = int(g["d"])
        lo = date(year, month, day) - timedelta(days=1)
        hi = date(year, month, day) + timedelta(days=1)
        return lo.strftime("%Y%m%d"), hi.strftime("%Y%m%d")
    m = _TOURNAMENT_RE.search(stem)
    if m:
        y = int(m.group("y"))
        return f"{y}0301", f"{y}0415"
    m = _SEASON_RE.search(stem)
    if m:
        y1 = int(m.group("y1"))
        return f"{y1}1101", f"{y1 + 1}0430"
    m = _BARE_YEAR_RE.search(stem)
    if m:  # "｜ 2025 ACC Men's Basketball" = that spring's games
        y = int(m.group("y"))
        return f"{y}0101", f"{y}0430"
    return None


def build_corpus_index(corpus_dir: Path, league: str) -> list[dict]:
    """One row per corpus summary: event_id, date, team locations, path.

    Cached at <corpus_dir>/_pairing_index_<league>.json; delete to rebuild.
    """
    cache = corpus_dir / f"_pairing_index_{league}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    rows: list[dict] = []
    for path in sorted((corpus_dir / league).glob("*/*.json")):
        try:
            summary = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue  # unreadable corpus file: skip, never abort the index
        comp = ((summary.get("header") or {}).get("competitions") or [{}])[0]
        teams = [c.get("team") or {} for c in comp.get("competitors") or []]
        locs = sorted(str(t.get("location", "")) for t in teams)
        # every exact name a title might use for each team
        names = {
            str(t.get("location", "")): sorted(
                {str(t[k]) for k in ("location", "displayName", "nickname",
                                     "abbreviation") if t.get(k)}
            )
            for t in teams
        }
        iso = str(comp.get("date", ""))
        day = iso[:10].replace("-", "")  # 2026-03-12T00:30Z -> 20260312
        if len(locs) == 2 and all(locs) and len(day) == 8:
            rows.append(
                {"event_id": str(comp.get("id", path.stem)), "date": day,
                 "locs": locs, "names": names, "path": str(path)}
            )
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows))
    tmp.rename(cache)
    return rows


def pair_videos_local(
    videos_dir: Path, corpus_dir: Path, out_dir: Path,
    league: str = "mens-college-basketball",
) -> dict:
    """Pair every mp4 against the local corpus; write pbp.json per unique
    match (persisted per item) + pairing_report.json with measured fractions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index = build_corpus_index(corpus_dir, league)
    # exact alt-name -> set of ESPN locations (location/displayName/nickname/
    # abbreviation collected at index time; collisions keep every location)
    name_to_locs: dict[str, set[str]] = {}
    events_by_loc: dict[str, list[dict]] = {}
    for row in index:
        for loc in row["locs"]:
            events_by_loc.setdefault(loc, []).append(row)
        for loc, names in row.get("names", {}).items():
            for n in names:
                name_to_locs.setdefault(n, set()).add(loc)

    def candidates_for(video_name: str) -> list[dict] | None:
        matchup = matchup_from_title(video_name)
        if matchup is None:
            return None
        locs_a = name_to_locs.get(matchup[0], set())
        locs_b = name_to_locs.get(matchup[1], set())
        found: dict[str, dict] = {}
        for la in locs_a:
            for row in events_by_loc.get(la, ()):
                other = row["locs"][0] if row["locs"][1] == la else row["locs"][1]
                if other in locs_b:
                    found[row["event_id"]] = row
        window = title_date_window(video_name)
        rows = list(found.values())
        if window:
            lo, hi = window
            rows = [c for c in rows if lo <= c["date"] <= hi]
        return sorted(rows, key=lambda c: c["date"])

    # a video already paired to an event claims it; rematch-ambiguous videos
    # resolve by elimination (e.g. the tournament re-meeting is claimed by the
    # tournament-titled video), iterated to a fixpoint
    claimed: set[str] = set()
    for existing in out_dir.glob("*.pbp.json"):
        try:
            comp = ((json.loads(existing.read_text()).get("header") or {})
                    .get("competitions") or [{}])[0]
            if comp.get("id"):
                claimed.add(str(comp["id"]))
        except (OSError, json.JSONDecodeError):
            continue

    report: dict = {"paired": [], "already": [], "ambiguous": [],
                    "no_match": [], "unparseable": []}
    pending: list[Path] = []
    for video in sorted(videos_dir.glob("*.mp4")):
        if pbp_path_for(video, out_dir).exists():
            report["already"].append(video.name)
        else:
            pending.append(video)

    def try_pair(video: Path) -> str:
        """-> 'paired' | 'ambiguous' | 'no_match' | 'unparseable'"""
        rows = candidates_for(video.name)
        if rows is None:
            report["unparseable"].append(video.name)
            return "unparseable"
        rows = [c for c in rows if c["event_id"] not in claimed]
        if not rows:
            report["no_match"].append(
                {"video": video.name,
                 "matchup": list(matchup_from_title(video.name) or ())}
            )
            return "no_match"
        if len(rows) > 1:
            return "ambiguous"  # not recorded yet: elimination may resolve it
        summary = json.loads(Path(rows[0]["path"]).read_text())
        if not summary.get("plays"):
            report["no_match"].append(
                {"video": video.name,
                 "reason": f"event {rows[0]['event_id']} has no plays"}
            )
            return "no_match"
        target = pbp_path_for(video, out_dir)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary))
        tmp.rename(target)
        claimed.add(rows[0]["event_id"])
        report["paired"].append(
            {"video": video.name, "event_id": rows[0]["event_id"],
             "date": rows[0]["date"], "plays": len(summary["plays"])}
        )
        print(f"OK {video.name} -> {rows[0]['event_id']}", flush=True)
        return "paired"

    while pending:
        progressed = False
        still: list[Path] = []
        for video in pending:
            outcome = try_pair(video)
            if outcome == "ambiguous":
                still.append(video)
            elif outcome == "paired":
                progressed = True
        pending = still
        if not progressed:
            break
    for video in pending:  # irreducibly ambiguous after elimination
        rows = [c for c in (candidates_for(video.name) or [])
                if c["event_id"] not in claimed]
        report["ambiguous"].append(
            {"video": video.name,
             "candidates": [(c["event_id"], c["date"]) for c in rows]}
        )

    total = sum(len(v) for v in report.values())
    ok = len(report["paired"]) + len(report["already"])
    report["totals"] = {
        "videos": total, "paired_or_existing": ok,
        "pairable_fraction": round(ok / total, 4) if total else None,
        "ambiguous": len(report["ambiguous"]),
        "no_match": len(report["no_match"]),
        "unparseable": len(report["unparseable"]),
    }
    (out_dir / "pairing_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["totals"], indent=2), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dates", help="YYYYMMDD-YYYYMMDD inclusive (network path)")
    src.add_argument("--pbp-corpus", type=Path,
                     help="season-pull corpus root (local pairing, Ask 0a)")
    ap.add_argument("--league", default="mens-college-basketball")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.pbp_corpus:
        pair_videos_local(args.videos_dir, args.pbp_corpus, args.out_dir,
                          league=args.league)
        return

    start_s, end_s = args.dates.split("-")
    start = date(int(start_s[:4]), int(start_s[4:6]), int(start_s[6:]))
    end = date(int(end_s[:4]), int(end_s[4:6]), int(end_s[6:]))
    dates = [
        (start + timedelta(days=i)).strftime("%Y%m%d")
        for i in range((end - start).days + 1)
    ]

    videos = sorted(args.videos_dir.glob("*.mp4"))
    pending = [v for v in videos if not pbp_path_for(v, args.out_dir).exists()]
    print(f"{len(videos)} videos, {len(pending)} missing PBP", flush=True)
    if not pending:
        return

    events = scoreboard_events(dates)
    print(f"scoreboard: {len(events)} ACC games across {dates[0]}..{dates[-1]}", flush=True)
    for video in pending:
        matchup = matchup_from_title(video.name)
        if matchup is None:
            print(f"SKIP (unparseable title): {video.name}", flush=True)
            continue
        event = events.get(frozenset(matchup))
        if event is None:
            print(f"NO MATCH on scoreboard: {video.name} -> {matchup}", flush=True)
            continue
        summary = fetch_summary(event["event_id"])
        n_plays = len(summary.get("plays") or [])
        if n_plays == 0:
            print(f"EMPTY plays for {event['name']} ({event['event_id']}) — not persisting", flush=True)
            continue
        target = pbp_path_for(video, args.out_dir)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary))
        tmp.rename(target)
        print(f"OK {event['name']} ({event['event_id']}): {n_plays} plays -> {target.name}", flush=True)


if __name__ == "__main__":
    main()
