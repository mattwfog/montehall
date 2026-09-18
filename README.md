# montehall

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
