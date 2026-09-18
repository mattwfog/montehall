# CV Brain — the thesis (why the program reordered)

The reasoning record behind `cv-brain-token-contract.md` (the mechanics).
The decisions below date from 2026-07-17→18; consequences cite what was
actually built and measured. The polished statement of the same argument
is `cv-state-estimation-paper.md`.

## 0. Where it came from

An audit of the program (07-17) found one recurring failure class:
capability celebrated without a wired consumer — engineering metrics
outrunning product truth. The corrective wasn't more caution; it was a
different center of gravity, ratified 07-18.

## 1. Vision is a primitive. The brain is the product.

RF-sensing research recovers body pose through walls from commodity
router signals — the information was never "in" the sensor; a learned
estimator with strong priors and a small closed world recovered it. Same
lesson as: ball visible in only ~12.7% of frames → full trajectory via
physics prior; humans who cannot read a single-frame jersey → perfect
identity via object permanence. **The product is a state estimator over
a closed world; every sensor — detectors, OCR, audio, score bug, PBP at
training time — is a swappable likelihood channel feeding it.**

Consequence: one token schema for all channels (`TOKENS_SCHEMA`), model
verdicts contractually banned from the stream, and the brain trained on
tokens, never pixels.

## 2. Ten men throwing a round circle inside a rectangle

The state is ~30 numbers (10 positions, ball + mode, clock, score,
possession) governed by known physics and rules that act as hard
constraints. The observation channel is an HD broadcast *designed for
human comprehension* — teams color-coded, score printed on screen, a
narrator saying names. Every term of the through-wall comparison runs in
our favor. The domain does not get to be hard; only a mis-specified
model of it does.

The two places difficulty actually lives — neither is the rectangle:

- **The editor's camera**: a moving crop serving the audience (5-7 of 10
  visible, replays spliced in). A nuisance handled with care (clock
  continuity distinguishes live from replay), not a wall.
- **The crossover kernel**: two bodies merge in occlusion and emerge —
  which is which? The single genuinely stochastic event, pairwise and
  bounded by the closed world. The sim channel manufactures unlimited
  labeled crossovers (built: `sim_traces.py`, 3 ft occlusion + p=.3
  observed-id swap).

## 3. The unit of measurement is not the frame

No human can read a number off one frame, and no human needs to. The
frame is the *sensor* unit; nothing forces it to be the *decision* unit
(the field defaults there via benchmarks + pipeline factorization). The
right unit: **dead-ball-to-dead-ball segment × the whole on-court
lineup, solved jointly** — subs only at whistles, one confident cue
names a whole segment, violations self-flag (two #24s on court).

The arithmetic that makes this mandatory: a wrong ID is not a missing
ID. Missing = one unknown row (abstention, benign). Wrong = population
contamination — it propagates through merges and displaces the rightful
owner. At 90% per-tracklet accuracy with 6 forced choices per line,
P(clean stat line) = 0.9⁶ ≈ 53%: aggregate metrics look good while most
lines are dirty. That is how a system can report 90%+ accuracy while
coaches distrust the sheet, and why the market leader ships human
annotators. Metrics must price the
asymmetry: **precision-at-coverage per stat line, abstention counted as
honesty** — and the entity-first / late-binding / coach-confirm design
(ruled 07-09) is the only structurally sound answer, not a UX
workaround.

## 4. External anchors beat internal confidence — every time it's been tested

Scoreboard oracle took made/missed from 40%/20% precision/recall to
10/10 where VLM confidence-math failed. PBP alignment produced .94
event recall where pixels-only produced 0.000. Roster constraints
killed the impossible-jersey class. Announcer audio — measured 07-18/19:
**94-189 exact roster-surname hits per game** — is the densest identity
anchor owned and was sitting unmined in every mp4. Estimated identity
loses; anchored identity wins. Anchor inventory: PBP (train-time), score
bug, rosters, name-calls, coach votes; faces are a *possible* future
anchor for top-division broadcast only, and are deliberately excluded
until there is an explicit biometric-privacy decision (BIPA, minors);
youth-sports incumbents also ship without face ID.

## 5. Tokens, not pixels — the economics that make iteration honest

A game is ~GB of video but ~KB-scale tokens. Measured: 12 epochs over
361 games ≈ **25 minutes on one DGX Spark**; a full brain iteration costs hours,
so hypotheses get tested instead of defended. Pixels are touched once,
by primitive extraction (the expensive part: full-stream extract ≈
3-4.5 h/game contended — rung economics live there, not in training).

## 6. Snap, not grind (the diagnosis rule)

Every decisive win in this system's history was a step-function when
structure landed (alignment 100% on game one; oracle 10/10; v3 .94 from
five mined games), and every grind was a mis-specified estimator being
tuned. **Standing rule: in this domain, an asymptotic grind is never
evidence the problem is hard — it is evidence the brain is missing
structure. Never add math to a grind.**

## 7. Universality posture

The harness (align film to an official feed, seal holdouts, one scorer)
and the brain (agents + ball + clock + score + closed rosters) are
sport-generic; primitives and truth feeds are per-sport config. Event
tier transfers ~1:1 (SoccerNet is the existence proof); identity
difficulty varies (soccer easier, hockey/football harder — and helmets
kill any face-founded identity, which is why the identity stack stays
jersey/body/closed-world/anchors). The moat is the align-and-supervise
engine, pointed at basketball first.

## 8. Snapshot as of 2026-07-19

- Substrate: 311 real games tokenized (sparse tier) + 50 sim games
  (dense truth incl. crossovers).
- Brain v0 (GRU, whole-game, multitask): sealed holdouts **.699/.500
  recall@5s** (Cal–FSU/Clemson–Duke) at v3-comparable precision, with
  ZERO real perception-channel training data — sim-to-real + anchors
  alone. v3 pipeline baseline = .94/.896. Made head at chance (no
  TV-board oracle yet — the scorebug reader remains that lever). Mode
  head learned sim dynamics (CE .159).
- Error attribution named the missing input — real perception channels —
  and rung-30 (top-30 games by aligned plays, extract → rich tokens →
  v0.1 retrain → holdout eval) was the first test of that hypothesis.
- **Tripwire (ratified): v0 line must snap to ≥.94 event recall within
  ~2 weeks of first training, or the token schema is missing structure.**

## 9. Open decisions

Biometric/face posture (top-division broadcast vs high school); rung scaling beyond
30; SSM swap timing (GRU is the placeholder latent-state model);
TV-graphics scorebug reader (unblocks made/missed + broadcast box
claims); adopting precision-at-coverage as the reported metric in every
CV summary — RESOLVED 07-31: adopted, see §10.

## 10. Ratified 2026-07-31: build the brain to its contract

Context (all code-verified): v0 implemented only the
event slice of the contract's §1 output space — a shot-probability
curve with three heads; no slot state, no permutation, no possession,
no segment boundary. The featurizer collapses per-player tokens to
team centroids and reduces jersey_read/name_call content to count
bumps, so the identity anchors of §4 had nothing to bind to. The
narrowing to an "event tier" happened at v0 implementation, was never
decided, and the contract tripwire's single event-recall scalar then steered two weeks of iteration (ladder
plateau .64–.74 vs .94). The tripwire's "token schema is missing
structure" arm named the wrong suspect: the 07-30 inverse ablation
acquitted the schema; the estimator was missing its contracted
structure. Independent confirmation the contract's shape is right:
SoccerNet 2026 PCBAS winners converged on per-player token streams +
joint event/identity sequence decoding (arXiv 2607.07320, 2606.09679)
— and their ~.59 Macro F1 ceiling confirms full-auto broadcast
attribution stays unshippable, i.e. §3's anchored/late-binding/coach
loop is the winnable posture.

The ruling (2026-07-31):
1. **The brain is built to contract §1** — the identity/state layer:
   per-entity streams survive featurization; output = per-slot roster
   distributions per dead-ball segment, abstention never forced.
2. **The v3 pipeline keeps the event job** until the brain's joint
   state beats it. No more re-deriving solved events.
3. **Headline metrics switch to the 07-09 set**: precision-at-coverage
   per stat line, coach-questions-to-complete, zero-fabrication.
   Event recall is a diagnostic, never the headline. The tripwire
   re-arms on identity metrics.
4. **The vote flywheel is the moat**: coach confirm/deny votes (collected
   in production since 07-10) become training anchors for the permutation
   model — the product improves with use.
5. **Ship gate (08-01): there is no product until this is
   solved.** Nothing ships unless the identity layer works. The 07-10
   stack is not "the product resuming in parallel" — it is
   infrastructure that waits for the solve.

## 11. Ratified direction 2026-07-31: coach-trainable annotations

An extension of the same decision: the vote loop generalizes beyond
identity — coaches can **add annotations beyond basic id** and thereby
**train their own models**. Coach-defined labels on video moments/
segments (scheme calls, custom events, whatever the coach tracks)
become supervision channels; a per-team personalization layer learns
them. This is the data-engine loop (§ research pass 07-10: Tesla
trigger→label→retrain) pointed at coach-owned vocabulary. In the
incumbents' products the humans are labor; here they are compounding
supervision. Token mechanics ride
the existing schema property ("sensors are policy, the schema doesn't
change" — contract §4).

Open cells (design decisions, not implementation details):
- Annotation vocabulary: typed palette vs free-form (vs both, staged).
- Per-team model form: per-team weights vs one team-conditioned model.
- Minimum-label threshold before a coach-trained concept renders.
- Privacy/sharing: private-per-team stands (07-09 ruling); cross-team
  pooling of anonymized annotation concepts UNRULED.
