# Trustworthy Player Statistics from Game Video via Closed-World State Estimation

## Abstract

Automated extraction of per-player statistics from basketball video is
commonly approached as a perception problem: detect objects per frame,
track them, and derive events heuristically from the tracks. We argue,
and show by measurement, that this factorization cannot produce
statistics a coach can trust, for a structural reason: identity errors
contaminate rather than degrade, so aggregate frame-level accuracy is
uninformative about stat-line correctness. We present an alternative
formulation — a belief-state estimator over a closed world, in which
every perception model is a swappable likelihood channel and decisions
are made at the possession-segment level with abstention permitted —
together with a supervision method that grades and trains the system
against official play-by-play feeds aligned to video time, eliminating
manual labeling for broadcast footage. We describe the evaluation
discipline the approach requires (single scorer, truth that no model
produced, sealed holdouts, per-component metrics across heterogeneous
footage classes), and report measurements in which externally anchored
estimation outperforms pixel-only pipelines by qualitative margins
(event recall 0.94 vs. 0.00 on identical holdout games), a
footage-agnostic mechanism outperforms its camera-specific counterpart
on the camera-specific footage itself, and a state estimator trained
purely on simulation and anchors recovers most event structure of real
broadcasts without any real perception training data. Current
component-level results across three footage classes are reported
honestly, including the unsolved ones.

## 1. Introduction

The target output of a game-video understanding system is small: a box
score is roughly thirty numbers per team, attributed to named players.
The input is large but highly structured: ten players, one ball, a
rectangle with known geometry, rules that constrain state transitions,
and — for broadcast footage — a feed designed for human comprehension
(color-coded teams, an on-screen clock and score, an announcer saying
names).

The dominant engineering approach factorizes the problem as
detect → track → classify events, with a hard identity decision made
per detection or per tracklet. We show in §2 that this factorization
fails not by degree but structurally, and in §3–4 that an estimator
built around the domain's closed-world structure, fed by external
anchors, both performs better and can be supervised at a scale no
labeling effort can reach.

## 2. The identity error model

The central observation is that **a wrong identity is not a missing
identity**. A missing attribution produces one unknown row and is
recoverable. A wrong attribution contaminates the population: the
event is credited to a player who did not perform it, propagates
through aggregation, and displaces the rightful owner's line.

The arithmetic is unforgiving. With per-decision accuracy p and k
forced identity decisions per stat line, the probability of a clean
line is p^k. At p = 0.90 and k = 6, P(clean) ≈ 0.53: a system whose
aggregate metrics read "90% accurate" delivers dirty lines half the
time. This explains a persistent industry pattern — systems claiming
90%+ accuracy whose output sheets practitioners do not trust, and a
market leader that ships human annotators rather than automation.

Two design consequences follow:

1. **Abstention must be permitted and priced as honesty.** The correct
   headline metric is precision-at-coverage per stat line, not
   aggregate accuracy over forced choices. A system naming 60% of
   lines at 99% precision is useful; one naming 100% at 90% is not.
2. **The frame is the sensor unit, not the decision unit.** No human
   can read a jersey number from a single frame, and no human needs
   to: substitutions occur only at dead balls, so one confident
   identity cue binds a player for an entire dead-ball-to-dead-ball
   segment, and impossible states (two of the same number on the
   floor) self-flag. The natural decision unit is the segment × the
   whole on-court lineup, solved jointly, with identity bound late —
   as evidence accumulates — rather than forced early.

## 3. Architecture: a belief-state estimator over a closed world

The system is organized as three persistent state stores — the ball
trajectory, a person registry (entities first; names bound late), and
game state (possession, period, attack direction) — updated over time.
Every perception model is a likelihood channel feeding these stores:
object detectors, a ball-specialist heatmap tracker, jersey OCR, court
segmentation and homography, and, at training time, aligned
play-by-play. Channels are swappable; the state contract is not.

Three principles govern the design:

**External anchors beat internal confidence.** Wherever a channel's
own confidence estimate competed with an external anchor, the anchor
won (§6): play-by-play alignment where pixel-only event detection
scored zero; a scoreboard oracle where vision-language confidence
arithmetic failed; roster constraints eliminating an entire class of
impossible jersey reads. The architecture therefore consumes anchors
natively — official feeds, the on-screen score bug, rosters, announcer
audio (measured at 94–189 exact roster-surname utterances per game),
and, in production, per-person confirmations by the end user, each of
which binds an entity for every event it owns.

**No footage assumptions in the architecture.** Any constant derived
from one camera setup is a latent defect for the next. A concrete
instance: a rim reference computed as the whole-clip median rim
position is only well-defined for a fixed camera; replacing it with a
local time-windowed reference (the median of the K nearest-in-time rim
detections) not only generalized to panning footage but improved
make/miss accuracy on the fixed-camera clip itself (5/8 → 7/8),
because the local reference also tracks slow drift. More generally:
constants in pixels assume a resolution, constants in frames assume a
frame rate, court dimensions assume a league. The discipline is to
work in object-relative units (rim widths, body heights) and time, and
to treat court geometry as an estimated input, never a constant.

**Tokens, not pixels, above the perception layer.** A game is
gigabytes of video but kilobytes of token streams. Pixels are touched
once, during primitive extraction; all estimation above trains and
iterates on tokens. Measured cost: 12 training epochs over 361
tokenized games complete in ≈25 minutes on a single workstation,
making hypothesis tests cheap enough to run instead of debate.

A meta-observation from the system's development history supports the
formulation: every decisive improvement arrived as a step function
when a structural element or anchor was added, and every extended
tuning effort ("grind") was later traced to a mis-specified estimator.
In a domain this constrained, an asymptotic grind is evidence of
missing structure, not of intrinsic difficulty.

## 4. Supervision at scale from aligned play-by-play

Manual grading of game footage yields tens of truth events per
person-session. The same information exists, for every broadcast game,
in the official play-by-play feed: every attempt, shooter, and outcome,
timestamped to the game clock. The alignment procedure is:

1. OCR the on-screen game clock densely across the video (a compact
   scene-text model over a localized crop).
2. Segment periods and validate monotonicity (the clock only counts
   down within a period; unconfirmed upward jumps are misreads).
3. Index (period, clock) → video time, and align each play to video
   time through its clock stamp.

This yields, per game, on the order of 100–350 aligned plays and
50–160 shot events with shooter identity and outcome — machine truth,
produced in minutes of compute, with no human labeling. Two secondary
products fall out for free: (a) spans without confident clock reads
are replays, cuts, or breaks, giving a live-play mask (an upper bound:
a cutaway that keeps the clock on screen still reads as live) used for
mining negatives and for honest coverage metrics; and (b) the roster
map (athlete → jersey) rides along from the feed's box score.

Timing caveat: a play's clock stamp lags the physical release by up to
~6 s, so truth files derived this way carry a wider matching tolerance
(±5 s, the action-spotting convention) than hand truth (±3 s).

The method defines a truth ladder by footage class: top-division
broadcasts have full PBP; lower divisions have equivalent feeds of
varying density; amateur footage without any feed falls back to
on-screen scoreboard self-labeling and box-score checksums, with hand
labels reserved for what nothing official covers. In our corpus, 384
broadcast games are fully aligned; converting one aligned game to a
scoreable truth file takes seconds.

## 5. Evaluation methodology

The measurement discipline is load-bearing; without it, camera
assumptions and self-confirming metrics accumulate silently. Five
rules:

1. **One scorer, one scorecard.** Every capability claim cites a
   scorecard entry by eval name; entries are written only by the
   scoring program. Unmeasured claims do not ship.
2. **Truth is never model-descended.** No system is graded on labels
   any model produced. Holdout games are sealed in code before any
   training touches the corpus.
3. **Acceptance spans footage classes.** A change is accepted only if
   its scorecard entries hold across all footage classes under test —
   currently a fixed-camera gym clip, a panning gym clip, and a
   broadcast segment. Single-class evaluation is precisely how
   camera-specific assumptions enter architectures unnoticed.
4. **Component metrics before composite metrics.** Ball, team
   identity, and player identity carry independent metrics; composite
   end-to-end numbers cannot localize a regression. Component metrics
   must also cover the whole film, not only event moments: a ball
   tracker in this system scored perfect recall measured at shot
   times while the ball was untracked for half the clip — a
   whole-film coverage metric (fraction of frames with a confirmed
   ball observation, longest gap, live-masked variant) exposed in one
   day what event-anchored metrics had structurally hidden.
5. **Error classes are eliminated by discrimination, not tuning.**
   Competing failure hypotheses are separated by targeted read-only
   probes before any parameter moves. Worked example: ball-coverage
   holes admit three causes — missing sensor supply, tracker gating
   discarding real observations, or no visible ball. Probing the raw
   channels inside the largest holes showed zero detector rows and
   heatmap peaks at the noise floor, exonerating the tracker. A second
   probe ran three detector checkpoints — the shipped fine-tune, a
   broadcast-trained fine-tune, and the stock COCO model — over the
   hole frames and over a control span where the shipped track was
   dense: all three found a ball on 0–1 of ~200 hole frames and on
   60–99% of the control frames, so no checkpoint was the fix. Viewing
   the hole frames closed the case: the broadcast hole is a director
   cutaway to the bench with the game clock running; the gym hole is a
   dead-ball stoppage with the clock frozen. Neither is a supply
   defect — no ball is on screen in either — and the clock-derived
   live-play mask (§4) counts the cutaway as live. The corrected lesson
   is the one rule 4 already states: a coverage metric's denominator
   must be ball-visible play, and a clock-derived live mask is not that.

## 6. Results

Measured comparisons in which structure or anchors were added while
all else was held fixed:

| comparison | without | with |
|---|---|---|
| broadcast event recall, sealed holdouts (pixel-only vs. PBP-aligned supervision) | 0.000 | 0.94 / 0.90 (two games) |
| made/missed precision/recall (VLM confidence arithmetic vs. scoreboard oracle) | 0.40 / 0.20 | 10/10 on scored deltas |
| impossible-jersey error class (unconstrained vs. roster-constrained) | present | eliminated |
| make/miss on fixed-camera clip (board-assisted vs. pure-CV geometric verdict) | 5/8 | 6/8 |
| make/miss, fixed clip (whole-clip median rim vs. footage-agnostic local rim) | 5/8 | 7/8 |

Sim-to-real evidence for the estimator formulation: a deliberately
plain sequence model (GRU over token streams, whole-game, multitask)
trained with **zero real perception-channel data** — simulation plus
anchors only — reached 0.699/0.500 event recall@5s on the two sealed
holdout broadcasts, against 0.94/0.896 for the fully engineered
pipeline. A placeholder estimator recovering the majority of real
event structure from simulated games alone indicates the information
resides in the domain's structure, which the estimator harvests;
error attribution identified its missing input as real perception
channels, not model capacity.

Current component-level state across the three footage classes (same
code, same scorer):

| metric | fixed gym (1080p) | panning gym (1080p) | broadcast (360p) |
|---|---|---|---|
| attempt detection recall / precision | .82 / .60 | .20 / .22 | .74 / .35 |
| ball recall at shot events | 1.0 | .20 | .35 |
| whole-film ball coverage | .48 | .75 | .51 (.50 live-masked) |
| make/miss accuracy | .875 | 1.0 (2/2) | .41 |
| shooter named (no roster, no scoreboard) | 2/5 | 0/2 | 0/17 |
| person consistency | 1.0 | — | 0 |

## 7. Limitations and open problems

- **Whole-film ball coverage** is 0.48–0.75 across classes, but the
  largest holes are not perception defects: they are cutaways and
  dead balls during which no ball is on screen (§5.5). The real
  bottleneck is therefore unmeasured until coverage is scored against
  a ball-visible-play denominator; the clock-derived live mask
  over-counts live play, and the gym clips have no live mask at all.
  Whether the residual holes during visible play are supply-limited
  is open.
- **Coverage measures presence, not correctness**: a track following
  the wrong object counts as covered. A correctness truth source
  (hand spot-checks or trajectory corroboration) is unbuilt.
- **Identity on low-resolution broadcast** is a measured zero and
  undiagnosed; candidate causes (body height below the OCR crop
  minimum at 360p, domain-shifted re-identification, camera cuts
  breaking tracklets) have not yet been discriminated.
- **Team assignment on broadcast** is structurally unscored pending a
  mapping between feed-side team labels (home/away) and the system's
  appearance clusters.
- Full-automation shooter attribution on broadcast has a low ceiling
  in the literature as well; the abstention/late-binding/confirmation
  design (§2) is chosen precisely because forced full automation is
  the losing posture.

## 8. Conclusion

Per-player statistics from game video is a state-estimation problem
over a small, rule-governed world observed through rich but
editorially framed sensors. Treating perception models as likelihood
channels around a persistent belief state, binding identity late at
the segment level with abstention permitted, supervising against
official feeds aligned to video time, and enforcing a measurement
discipline that separates component from composite and truth from
model output — each of these choices is supported here by direct
measurement against its simpler alternative. The mechanisms will
continue to change; the formulation and the measurement discipline
are the durable contribution.
