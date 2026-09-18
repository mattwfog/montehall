# Montehall CV — identity design-of-record: entity-first stat sheet + coach-in-the-loop binding

Status: **ratified 2026-07-10** — §3 principles and §5 staging approved.
Decided the same day: unnamed rows present as "Player A"-style labels,
plus a position descriptor when confidently derivable from court
positions; coach votes persist in their own table, not as JSON edits.
The questions in §7 stay open until their phase arrives.

This document captures the design decisions plus research-grounded
refinements; recommendations are marked REC and are proposals, not
decisions.

## 1. The reframe

Jersey identification is **not** the product metric, and comparing
jersey-read counts between pipelines concedes the wrong frame. The product
does **all analysis on anonymous entities**; identity is a labeling layer
applied late — by accumulating evidence over time, by a binding model, and
ultimately by the coach, who can name any player on sight.

Consequences:

- **No stat line is ever discarded for lack of a name.** Today
  `run_boxscore`'s roster filter is removal-only: stats whose entity can't
  bind to a roster number degrade into the team line. Under this design,
  unbound entities keep first-class stat rows, awaiting a label.
- **The machine's job is to ask few enough questions**, not to reach
  full-auto OCR. Published full-auto ceiling is 87–91% tracklet-level
  identification (Koshkina CVPR'24).
- After the coach identifies the residual entities — in-flow, while
  watching film — the coach gets **the same complete results**
  OCR-perfection would have produced.

## 2. What the leaders do (research pass, 2026-07-10)

Every leader in this market ships identity with a human in the loop or
hardware on the body. Nobody ships full-auto. Our design is the market's
own pattern, moved to the cheapest possible moment (in-video, in-flow)
and made compounding (the dossier).

**Sports incumbents:**

- **Hudl Assist** (market leader, human-analyst product): identity =
  jersey numbers matched against the roster the coach submits. When a
  number is unreadable or the player isn't on the submitted roster, the
  analyst tags **unknown**, and the coach completes identity through a
  built-in **"Identify Unknown Athletes" workflow**. Hudl also invites
  coaches to submit "distinguishing features" hints up front for
  obstructed jerseys. ([Assist FAQ](https://www.hudl.com/products/assist/faq),
  [support: breakdowns](https://support.hudl.com/s/article/basketball-breakdowns-and-reports-hudl-v3))
  → The market leader already ships coach-completes-identity as a normal,
  accepted workflow. Our design automates the analyst, keeps the workflow.
- **Veo Player Spotlight**: detects shirt numbers, then the coach
  **"assigns your lineup by linking your players to the detected jersey
  numbers"** — an explicit detected-entity→roster binding step. Veo
  publishes honest accuracy caveats: numbers 1–25 read better than 26–99;
  duplicate numbers confuse the system.
  ([Veo help](https://support.veo.co/hc/en-us/articles/27698176364817-Overview-of-Player-Spotlight-How-to-track-and-follow-players-in-your-Veo-recordings))
- **Trace** (amateur soccer): solves identity with **hardware** (GPS
  sensors in socks) — and still ships a deny flow ("Halo on the wrong
  player") whose reports "are used to improve future highlights."
  ([Trace support](https://support.traceup.com/content/processing-faq-playerfocus))
  → Even the hardware answer keeps a deny loop and feeds it back to
  training. Validates deny-as-training-signal.
- **Balltime (Hudl's volleyball AI)**: AI tags players; an **edit tool**
  lets users change misidentified athlete tags post-delivery.
  ([Balltime FAQs](https://academy.balltime.com/getting-started/faqs))
- **SportsVisio** (direct competitor): multi-cue identity continuity —
  "jersey numbers, team color, height, and motion patterns" — and claims
  92%+ attribution. Their own honesty caveat is instructive: **aggregate
  accuracy looks higher than play-by-play accuracy because errors
  cancel** — a 22-point box score can be 21 right attributions plus one
  wrong one. ([SportsVisio technical guide](https://www.sportsvisio.com/stories/how-ai-basketball-analysis-works))
  → We refuse that trade (0-fabrication stays a headline metric);
  confidence-gated exposure is the differentiator.
- **Pro tier (Stats Perform AutoStats / Second Spectrum / SoccerNet GSR)**:
  broadcast tracking fuses positions + jersey reads + ReID embeddings +
  team labels in post-processing to repair fragmentation and identity
  swaps ([GSR SOTA paper](https://arxiv.org/html/2504.06357v1)); a USPTO
  patent for camera-based sports tracking describes **signaling a human
  operator exactly when identity ambiguity occurs**. Pro identity is
  multi-camera + operators — not a bar amateur single-cam full-auto can
  clear, which is why the coach is the right operator in our tier.

**Consumer precedent — the strongest one:**

- **Google Photos face groups** is entity-first late binding at
  billions-user scale: faces cluster anonymously; the user **names a
  cluster once**; the system then suggests merges with a **three-way
  answer: Same / Different / Not sure**; labels can also drive merges
  ("label two groups the same name → merge?"). Grouping deliberately uses
  **non-face evidence** (same clothing, photos taken close in time) when
  the face isn't visible — the exact analog of our ReID/role/position
  cues. Two design lessons carried into §3: their **merges are
  irreversible** (documented user pain — ours must be revisable), and the
  feature succeeds at ~80–85% cluster accuracy **because correction is
  cheap** ([Google Photos help](https://support.google.com/photos/answer/6128838),
  [Luxand explainer](https://luxand.cloud/face-recognition-blog/how-does-google-photos-recognize-the-names-and-faces)).
  Face groups/labels are **private by default** — maps to our
  privacy posture for minor-athlete data.

**Interaction + data-engine precedent:**

- **Waze**: reports are confirmed by drivers **already passing the
  location** — proximity-triggered popup, lightweight "say thanks" /
  "not there", multiple "not there" votes remove the event, duplicate
  reports merge, reputation/gamification sustains contribution. Their
  documented UX failure: an **ambiguously-worded binary prompt** ("missing
  sign — still there?" …the sign or the problem?). Prompt wording must be
  unambiguous ([MorelandConnect UX critique](https://morelandconnect.com/ux-failure-waze/),
  [trust mechanics](https://arxiv.org/pdf/2511.03016)).
- **Tesla's data engine**: the canonical compounding loop — trigger
  signals (detection flicker, model uncertainty, cross-model
  disagreement) select which fleet examples humans label; labels retrain;
  models redeploy; the loop mines the long tail. The moat is the loop,
  not any one model. Caveat from the literature: raw uncertainty sampling
  picks redundant/outlier examples — production systems combine signals
  ([Mindkosh on Tesla's active learning](https://mindkosh.com/blog/how-tesla-uses-active-learning-to-elevate-its-ml-systems/)).

## 3. Crystallized approach

**One sentence:** Track anonymous entities perfectly, attribute every stat
to an entity, bind names late from accumulated evidence, and let the coach
resolve the remainder with one-tap confirmations while watching film —
each answer permanently teaching the system that team.

**The loop:**

1. Perception attributes stats to **entities** (exists — REJOIN2 Clemson:
   12/14 attempts entity-attributed).
2. The **binding layer** names entities whose evidence clears the bar
   (jersey posteriors × roster today; dossier adjudicator later).
3. Unnamed-but-stat-carrying entities become **questions**, asked
   Waze-style during film review: box on the player, "Is this #24 —
   Marcus?", confirm / not him / not sure.
4. Answers **re-key the stat sheet immediately** and persist as per-team
   **dossier anchors** (ReID embedding + name), so the same player
   auto-binds next upload.
5. Every answer is also a **verified training example** (real jersey
   crops = the scarcest asset; 43 exist today) — models improve, question
   count decays toward zero. Tesla's loop, with the coach as the oracle
   riding behavior they already have (film sessions).

**Design principles distilled from the research (REC — ratified as a set):**

1. **Ask in context** (Waze): prompts fire only while the entity is
   on-screen during playback — never as a separate chore screen (though a
   summary list view can exist as a fallback, Hudl-style).
2. **Three-way answers** (Google Photos): confirm / not-him / not-sure.
   "Not sure" is signal too (hard example), and prevents forced errors.
3. **Unambiguous prompt wording** (Waze's documented failure): the
   question names the player and shows the box — "Is this Marcus (#24)?"
   — never an ambiguous referent.
4. **Bindings are revisable, never irreversible** (Google Photos' merge
   pain): a binding is a high-confidence posterior, not a hard write;
   coach can re-open any row.
5. **Deny is a first-class diagnostic**: "not him" on a fused entity
   localizes a *tracking* error (split-flag), the only ground-truth QA
   source the entity layer has. Trace ships exactly this and feeds it to
   training.
6. **Multi-cue dossiers** (GSR/SportsVisio/Google's clothing signal):
   identity evidence = jersey posteriors + ReID + team + role/position
   patterns + coach answers; no single cue is load-bearing.
7. **Question selection is active learning with combined signals** (Tesla
   caveat): ask about entities by (stat-weight × evidence-uncertainty ×
   on-screen quality), capped by a per-session budget — not raw
   uncertainty order.
8. **Never let errors cancel into fake aggregate accuracy** (SportsVisio's
   caveat): confidence-gated exposure stays; 0-fabrication
   stays a headline metric.
9. **Roster is submission context** (Hudl + Veo both require it): the
   roster.json upload path (live in prod) is the binding vocabulary;
   prompts offer roster names, not free text.
10. **Identity data is private per team by default** (Google Photos
    posture; for minor athletes the dossier is biometric-adjacent data
    with retention/deletion obligations).

**Metrics (supersedes jersey-count reporting):**

1. **Entity-attribution coverage** — % of stat events attributed to a
   specific entity (Clemson 86% vs 3 auto-named rows shows the naming gap
   ≠ perception gap).
2. **Coach questions to complete** — target: decays per game as the
   dossier accumulates; the Waze/Tesla-style compounding KPI.
3. **0 fabricated rows** — headline, always.

## 4. Architecture (what exists vs what this adds)

| Layer | State |
|---|---|
| Anonymous entity tracking (track_ids → merged entities; jersey used only as merge veto) | **EXISTS** — extract → team_assoc, long-gap ReID merges |
| Per-entity primitives (crops, court positions, ball controls, jersey-read posteriors, ReID embeddings, shot/rebound involvement) | **EXISTS** — artifact store per job |
| Entity-level attribution (shooter, rebounds) | **EXISTS** — `box_score` rows carry `entity_id`; `shooter_attributed` counts entities |
| Late roster binding (posterior × roster, margin + ambiguity guard) | **EXISTS** — removal-only; unbound collapses to team line |
| **Entity rows first-class in the box score** (unbound = "awaiting label", never folded away) | **NEW** — keying change in `run_boxscore` + app transformer |
| **Identity adjudicator over the full dossier** (reads an entity's whole evidence trail, pronounces identity + confidence; revisable) | **NEW** — binding today is deterministic rules; the possession harness is the pattern, pointed at identity. Natural evidence format = the VLM contact-sheet (tiled best-N legible crops + roster context) |
| **Coach-in-the-loop binding — Waze-style in-video confirm/deny** (video plays, box appears, "Is this Marcus (#24)?" — confirm / not him / not sure) | **NEW** — per-frame boxes exist in detections artifacts; play payloads carry ≤4Hz trajectories; gap = overlay data delivery to the SPA + prompt component + identity-vote API. Hudl-style "identify unknowns" list view as the non-playback fallback |
| **Cross-game dossier persistence** (per-team: coach answers + ReID anchors; confirmed player auto-binds next upload) | **NEW** — today every job is independent. Research phase flagged identity posteriors/evidence trails as no-prior-art territory |
| Flywheel: every answer = (a) immediate stat label, (b) dossier/ReID anchor, (c) verified training crop | **NEW** — automates what the roster harvester hand-cranks (43 verified crops to date) |

Assessed-not-ruled sidebars: VLM contact-sheet reads (cheap aggregated-view
OCR upgrade + adjudicator evidence format — testable against the 148-read
zero-shot baseline on the UConn clip); Gaussian-splat jersey reconstruction
**parked** (single fixed cam + non-rigid subjects + low-res crops = the
most expensive aggregator for no added signal).

## 5. Staged path (REC)

- **Stage A — entity-first sheet + post-hoc labeling (Hudl parity):**
  `run_boxscore` keys entity-else-jersey; app shows unnamed rows with best
  crops; coach labels from a list view; answers re-key the sheet. No new
  perception. This alone converts Clemson's 12/14 attribution into a
  nameable sheet.
- **Stage B — in-video Waze prompts + dossier persistence:** overlay boxes
  during playback, three-way prompts under a budget; per-team dossier
  table; confirmed anchors auto-bind on the next upload. This is the
  compounding moat.
- **Stage C — adjudicator + active-learning tuning:** dossier-reading
  adjudicator (contact-sheet VLM or rules-v2) pre-answers questions;
  question selection tuned on measured coach behavior; OCR/ReID retrains
  ride the minted labels.

Each stage is independently shippable and coach-visible; none blocks the
others' perception work (detector/ReID/OCR improvements keep landing
underneath).

## 6. Verification bar (plumbing ≠ shipping)

The feature is done only when: a real uploaded game produces an entity-first
stat sheet in the app; the coach answers K prompts; the sheet re-keys to
fully-named rows; and on a subsequent upload of the same team, a
previously-confirmed player auto-binds with zero prompts. A test that severs
the vote→re-key wiring must fail.

## 6a. Implementation judgment log (2026-07-10)

- **0.3 result:** contact-sheet VLM reads on the UConn clip = **264
  accepted / 253 roster-legal of 313 tracklets vs 19-tracklet per-frame
  baseline** (~$0.60 of model calls). Aggregated-view reading wins decisively.
- **C1 = suggestion-only** (principle 9): contact reads surface as
  `suggested_number` pre-fill on unnamed rows; they never name a row and
  never enter roster binding — posterior fusion into `run_jersey` waits
  for an eval against verified labels (no silent behavior change to the
  0-fabrication path).
- **B2 mechanics:** three-way (Yes / Not them / Not sure); prompt only
  while the entity is on-screen ≥6% of frame height; 5 prompts per
  session; deny routes to the naming modal (roster picker) and is recorded
  (tracking-QA signal); wording always names the suggestion.
- **B3 = append-only `cv_player_anchors`** minted by every confirm
  (team-scoped, CASCADE deletion paths, conservative retention pending
  the privacy decision).
  **ReID-gallery auto-bind (embedding match across uploads) is NOT built —
  named as the next arc, not silently skipped**; today anchors are the
  durable dossier + decay metric substrate.
- **A6 evidence = contact-sheet JPEGs reused** as the modal's "who is
  this?" image; keys ride the results JSON, api presigns.

## 6b. Broadcast + PBP direction (2026-07-10)

Direction: broadcast games first evaluate a baseline, and then become the
substrate for building custom models. The PBP-matched broadcast games serve
the arc in sequence: (1) establish the objective baseline — play-by-play
truth (ESPN per game) scores the current pipeline's
box-score output; (2) then the same PBP↔video alignment becomes the
substrate for building our own custom models — event windows, verified
shooter/jersey labels at scale, made/missed distillation targets.
First build item: score-bug clock OCR → game-clock↔frame alignment;
probe game = Clemson–Duke.

## 7. Open design questions

1. **Box-score presentation** of unnamed entity rows (naming — "Player A"
   vs number-guess-with-badge; ordering; whether team lines remain).
2. **Adjudicator form** — rules-v2 vs LLM dossier reader; end-of-run only
   vs deferred re-adjudication as evidence grows.
3. **Prompt budget/pacing** — max prompts per session; stat-carrying
   entities only; on-screen size/quality gate.
4. **Deny semantics** — "not him" → next hypothesis vs roster picker;
   split-flag ("that's two different players") placement.
5. **Spot-check confirmations** of high-confidence bindings — ever shown?
6. **Overlay delivery** — box data format to the SPA; which player UI
   surface hosts it.
7. **Dossier persistence** — schema, per-team scope, storage (PG vs
   artifact store), retention/deletion. Must be decided together with
   the privacy requirements for minor-athlete data (per-subject
   deletion).
8. **OCR pre-fill** — prompts carry a suggested name (Google-style) vs ask
   open (higher friction, less anchoring bias).
9. **Rollout order** — relative to the parked runner-placement decision.
