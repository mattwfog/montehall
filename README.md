# montehall

[![tests](https://github.com/mattwfog/montehall/actions/workflows/tests.yml/badge.svg)](https://github.com/mattwfog/montehall/actions/workflows/tests.yml)

Computer vision for team sports, built around one idea: **vision is a sensor, not
the answer**. Detectors, trackers, OCR and calibration produce observations; a
separate layer reconstructs game state from them; learned models reason over that
state. Identity is bound late, from accumulated evidence, and every capability
claim has to cite a measured result.

The argument, with measurements, is in
[Trustworthy Player Statistics from Game Video via Closed-World State Estimation](basketball/docs/cv-state-estimation-paper.md).

The repository holds two independent Python projects that apply that idea to two
sports.

| | [`basketball/`](basketball/) | [`soccer/`](soccer/) |
| --- | --- | --- |
| Package | `montehall_cv` | `soccerviz` |
| Input | fixed-camera and broadcast game film | broadcast video, provider tracking and event data |
| Core | entity-first tracklets, ball spine, person registry, shot attribution, box score | detection, pitch calibration, tracking, possession and tactics models, durable specialist harness |
| Evaluation | one scorer writes [`SCORECARD.json`](basketball/SCORECARD.json) | frozen benchmarks with tracked JSON reports in [`results/`](soccer/results/) |
| Start here | [`basketball/README.md`](basketball/README.md), [`docs/cv-brain-thesis.md`](basketball/docs/cv-brain-thesis.md) | [`soccer/README.md`](soccer/README.md), [`docs/STATUS.md`](soccer/docs/STATUS.md) |

## Shared design commitments

- **Observations, state, model.** Per-frame perception is kept separate from
  reconstructed game state, and from the models trained over that state.
- **Late-bound identity.** A track is the persistent object. Team, jersey number
  and player name are hypotheses with evidence, never stored primitives.
- **Measured claims only.** Comparisons cite a scorecard entry or a checked-in
  results file. Negative results and withdrawn recommendations stay in the record.
- **Permissive licensing by default.** Components are chosen for MIT, Apache or
  BSD terms. Exceptions are optional and listed in [`NOTICES.md`](NOTICES.md).
- **Resumable stages.** Long runs write incrementally and skip completed work.

## Models as likelihood channels

The reasoning layer follows the same rule as perception: a model is a swappable
channel, and its confidence has to be measured before it drives a decision. The
basketball possession harness ([`harness/adjudicators.py`](basketball/montehall_cv/harness/adjudicators.py))
ships two backends behind one interface:

- `haiku` — one generative call per possession; the model writes a JSON verdict and
  reports its own confidence.
- `jev` — [TypeSafe's](https://docs.typesafe.ai) System One model, which generates no
  text. Code asks typed questions over the possession trace (how did it end, who
  scored, does the last pass pass the FIBA assist test, does the trace contradict
  itself) in a single request and composes the verdict from the returned
  probabilities. Abstention is a threshold, and an uncertain scorer stays a team
  stat instead of a guessed player.

[`harness/calibration.py`](basketball/montehall_cv/harness/calibration.py) scores any
backend's confidence (expected calibration error, Brier score, reliability bins,
precision-at-coverage), and `python -m montehall_cv.harness.compare` runs backends
side by side on the same traces against truth no model produced. The backends and
the scoring are tested, including a round trip through the real TypeSafe client
with the HTTP transport mocked.

A live probe ([`scripts/jev_probe.py`](basketball/scripts/jev_probe.py), six hand-built
traces, `jev-1.13.0`, 2026-09-18) is a behaviour check, not a benchmark. The model
answered all four questions for a possession in about 0.4 s and roughly 1,000 input
tokens, and its probabilities were graded rather than saturated: a clean make 1.00,
a miss followed by defensive control 0.98, a no-shot change of possession 0.81, and
a one-sample trace came back `unclear` at 0.64, so the verdict abstained. It also
failed one case: given a made shot in a possession where only the defense ever held
the ball, its self-contradiction judgment was 0.13. That contradiction is computable,
so it is now detected in code (`structural_anomalies`) and the model is only asked
for what code cannot decide.

### Scored on simulated possessions

The state simulator knows how every possession ended, so it supplies exact truth.
[`harness/sim_possessions.py`](basketball/montehall_cv/harness/sim_possessions.py) replays
sim games and hands the adjudicator the degraded trace the pipeline would see: ball
control only on ticks the ball was observed, tracker id swaps included, shots detected
82% of the time and their made flag right 87.5% of the time (the fixed-camera figures
from the paper). 300 possessions, seed 11, `jev-1.13.0`, 2026-09-18
([`results/adjudicators-sim.json`](basketball/results/adjudicators-sim.json)):

| adjudicator | coverage | outcome accuracy | mean confidence | ECE | scorer precision / coverage |
|---|---|---|---|---|---|
| no model: trust the last shot flag | 1.00 | 0.663 | n/a | n/a | 0.71 / 0.73 |
| jev | 0.79 | 0.646 | 0.905 | 0.260 | 0.70 / 0.71 |
| jev, told the sensors' error rates | 0.79 | 0.646 | 0.872 | 0.227 | 0.71 / 0.70 |

The result is negative and it replicated on a second seed. The model does not beat a
one-line rule, and it is overconfident: verdicts it reports at 0.97 are right 67% of the
time, so raising the threshold buys little precision (0.65 at full coverage, 0.70 at
half). It reads the made flag as fact. Putting the measured error rates in its state
moved calibration only slightly. The information limit here is in the sensors, not the
adjudicator, which is the thesis's own argument: a better shot channel or an external
anchor will move this number and a smarter reader of the same evidence will not. The
model's raw confidence should not drive abstention without recalibration against truth.
Limits: simulated possessions and stated noise, not real footage; the generative
backend has not been scored on this set (`python -m montehall_cv.harness.sim_eval
--backends haiku jev` runs it).

## Running the tests

Each project is a standalone [uv](https://docs.astral.sh/uv/) project with its own
lockfile.

```sh
cd basketball && uv sync && uv run pytest -q
cd soccer && uv sync && uv run pytest -q
```

GPU work ran on an NVIDIA DGX Spark (GB10, aarch64) inside NGC PyTorch containers.
The base installs and the test suites do not need a GPU. Video, datasets and model
weights are not included in this repository.

## License

Apache-2.0, see [`LICENSE`](LICENSE). Third-party components and datasets are
listed in [`NOTICES.md`](NOTICES.md).
