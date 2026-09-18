# UnravelSports formation experiment

The real UnravelSports EFPI package now runs locally against SoccerViz provider tracking.
This is a working optional CPU worker with saved assignments and reproducible inputs.
It is **not a validated formation recognizer or a replacement for the existing baseline**.
No training or detector fine-tuning is involved in this experiment.

## Executed results

Both runs use the first five minutes of the existing Metrica games, sampled at 1 Hz
(source timestamps 0.04 through 299.04 seconds). All 300 sampled frames per game
contain 11 observed players on each team. Additional eligibility checks retain only
frames with an observed ball, on-pitch finite player coordinates, and an existing
provider-event possession interval. No missing player, ball, or possession is filled.

| Measure | Game 1 | Game 2 |
|---|---:|---:|
| Sampled frames | 300 | 300 |
| Both teams complete | 300 / 300 | 300 / 300 |
| EFPI eligible | 126 / 300 (42.0%) | 209 / 300 (69.7%) |
| Omitted | 174 | 91 |
| Missing provider-event possession interval | 171 | 84 |
| Ball unavailable | 130 | 51 |
| Home player outside assumed pitch bounds | 36 | 12 |
| Away player outside assumed pitch bounds | 24 | 5 |

Omission reasons overlap. Out-of-bounds observations are omitted rather than clamped;
the 105 × 68 m dimensions are an existing project assumption, not verified stadium dimensions.
The eligibility Parquet records every sampled frame and every omission reason.

| Game / team | Dominant frame EFPI template | Share | Same-phase adjacent stability | Existing baseline stability |
|---|---|---:|---:|---:|
| 1 / home | 442 | 28.6% | 67.6% | 86.7% |
| 1 / away | 1234 | 13.5% | 61.0% | 82.9% |
| 2 / home | 442 | 13.4% | 60.2% | 82.7% |
| 2 / away | 442 | 24.9% | 68.1% | 87.4% |

Stability means unchanged template across consecutive eligible 1-second samples with
the same event-derived possessing team. It does not bridge omitted frames or possession
changes. There are 105 comparable pairs per team in game 1 and 191 in game 2.
`1234` is the upstream template's name, not an assertion that an analyst would label
the nominal team formation 1-2-3-4. EFPI can match unconventional transient arrangements.

Name agreement with the existing three-template baseline is 17.1% and 17.9%.
This is **not accuracy**: the template sets and scaling differ, and the baseline picks
the deepest player as its keeper independently in each frame. EFPI here uses a fixed
keeper proxy. Broader template choices naturally allow more label changes.

EFPI also ran its actual `every="1m"` implementation. These fits average available
positions separately by team-in-possession within each minute; they are not contiguous
possession segments. Saved minute assignments should be reviewed before choosing a
smoothing or template-selection policy. Their windows have varying observed support.

The practical recommendation is to retain EFPI as an optional analysis comparator,
review minute fits with analyst labels, and choose an appropriate template subset and
minimum window coverage before exposing confident formation names in the product.
The unconstrained frame-level output is too changeable to promote on these results.

## Data fidelity and assumptions

- The existing Kloppy provider importer preserves CSV frame IDs, exact CSV clocks,
  team membership, and missing observations. Both home and away source hashes are saved.
- Goalkeepers are **unverified fixed geometry proxies**, selected per segment as the
  sufficiently observed player with smallest median distance to either goal. A minimum
  3 m separation from the next player is required. The resulting team defended goals
  must be opposite. In both games the selected IDs were `home_11` and `away_25`.
- Game 1's home team is inferred to attack toward increasing source x; game 2's home
  team toward decreasing source x. The source coordinate system is top-left metres.
  The adapter converts to centred Cartesian coordinates, then static home-right, and
  lets Unravel orient to the explicitly supplied event-derived possessing team.
- The worker supplies the existing retrospective `possessions.parquet` intervals.
  It omits gaps and does not allow Unravel's default nearest-ball inference to create
  missing possession labels. Possession file and raw event file hashes are saved.
- Actual `KloppyPolarsDataset` conversion is used with smoothing disabled and explicit
  role metadata. Its output is checked row by row against the expected source transforms:
  **7,705 rows verified; maximum x/y or timestamp error 0** across both final runs.
- Metrica CSV supplies no ball height. `z=0` is an API placeholder; EFPI only uses x/y.
  No spatial missingness is interpolated. This experiment does not measure pressing,
  nominal tactical roles, or formation accuracy against ground truth.

## Reproduce

The worker uses `.venvs/formations`, separate from core and analysis environments.
It requires the existing raw Metrica files and processed possession tables.

```sh
uv venv .venvs/formations --python 3.12
uv pip install --python .venvs/formations/bin/python -r envs/formations/requirements.txt
MPLCONFIGDIR=/private/tmp/soccerviz-formation-mpl PYTHONPATH=src \
  .venvs/formations/bin/python -m soccerviz.candidates.formation \
  --game 1 --out artifacts/integrations/formation-metrica-game-1
MPLCONFIGDIR=/private/tmp/soccerviz-formation-mpl PYTHONPATH=src \
  .venvs/formations/bin/python -m soccerviz.candidates.formation \
  --game 2 --out artifacts/integrations/formation-metrica-game-2
MPLCONFIGDIR=/private/tmp/soccerviz-formation-mpl PYTHONPATH=src \
  .venvs/formations/bin/python -m pytest tests/test_formation_adapter.py -q
```

Use a new output directory for repeat runs: existing outputs are never overwritten.
The pinned worker was executed with Python 3.12.12, UnravelSports 1.2.1,
Kloppy 3.19.0, mplsoccer 1.8.0 and Polars 1.44.1 on macOS arm64. CPU processing
after imports took about 2 seconds per game; that is not a full pipeline throughput
benchmark. Initial package imports and installation are outside that timing.

Ten focused tests pass, including an actual EFPI invocation, preservation of source
clocks and axes, missing-player exclusion, cross-team identity rejection, malformed
coordinates, missing ball/possession, and stability across gaps. Upstream Polars
deprecation warnings are recorded; no dependency or source modifications were made
inside UnravelSports.

Final run directories:

- `artifacts/integrations/formation-metrica-game-1/`
- `artifacts/integrations/formation-metrica-game-2/`

Each contains `report.json`, raw sampled `source-observations.parquet`,
`frame-eligibility.parquet`, verified `efpi-input.parquet`, `frame-assignments.parquet`,
`minute-assignments.parquet`, `minute-segments.parquet`, baseline assignments and
an artifact SHA-256 manifest. Reports record source, adapter, baseline and installed
EFPI implementation hashes plus package versions. Earlier `formation-game-*` and
`formation-smoke-*` directories are development runs; use the final `formation-metrica-*`
directories for reproducible results.

## Primary references

The implementation calls the public `EFPI(dataset=...).fit(every=...)` API documented
by [UnravelSports](https://github.com/UnravelSports/unravelsports). EFPI is a template
matching and assignment method; see [Bekkers, EFPI (2025)](https://arxiv.org/abs/2506.23843).
UnravelSports is distributed under MPL-2.0, and its installed sources are unmodified.
The observations come from [Metrica sample data](https://github.com/metrica-sports/sample-data)
at the revision already pinned by this repository. Those upstream descriptions support
the method choice; all numerical results above come from the saved local executions.
