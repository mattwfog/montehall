# Public StatsBomb training and evaluation

The expanded baseline uses 30 real UEFA Euro 2020 matches from [StatsBomb Open Data](https://github.com/hudl/open-data), pinned to commit `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`. The importer downloads the official events, lineups, competition match catalogue, README and license. Every source has a URL, SHA-256 and byte count in the manifest. Data attribution: **StatsBomb**. The downloaded `LICENSE.pdf` retains the upstream terms; availability is not an assertion of unrestricted commercial licensing.

The deterministic split is committed to `splits.json` **before downloading event outcomes**. Match IDs are ordered by `sha256("soccerviz-v1:" + match_id)`, then selected and divided into 20 training, 5 development and 5 test games. These are our research splits, not a claimed official StatsBomb benchmark. Team identities can recur across splits. No action-level random split is used.

| Partition | Matches | SPADL actions |
|---|---:|---:|
| Train | 20 | 45,403 |
| Development | 5 | 10,854 |
| Test | 5 | 11,612 |

The official socceraction 1.5.3 local StatsBomb loader converts events into SPADL, then normalizes every action into attacking-left-to-right coordinates. Period 5 shootouts are excluded. The training worker verifies split hashes, action-file hashes, exact game membership, disjointness and orientation before fitting.

## What was trained

**xT:** socceraction's 16×12 ExpectedThreat grid, fitted only on the 20 training matches and used to rate successful movements in the five test matches. This confirms the fit/rate pipeline; it does not measure xT decision quality.

**VAEP probability baseline:** separate standardized logistic classifiers for scoring and conceding. Features describe the current completed action: action type, outcome, body part, start/end positions, polar positions and movement. No player/team/game IDs are model features. This is a deliberately modest baseline, not a reproduction of the full three-action VAEP architecture. Targets use socceraction's next-ten-action labels, including the current action, confined to each match and period. The horizon truncates at period ends.

Each classifier is fitted on train only with `C` in `{0.1, 1.0}`. The development set selects the smaller log loss; ties select the smaller `C`. The selected model and protocol are written before test targets and metrics are constructed. No refit on dev and no test tuning occurred. Model coefficients and scaler parameters are saved as an NPZ file with feature names and a content hash; inference does not require loading a pickle.

## Held-out results

| Target | Model Brier ↓ | Train-prevalence Brier ↓ | Model log loss ↓ | Train-prevalence log loss ↓ |
|---|---:|---:|---:|---:|
| Score | 0.013189 | 0.014876 | 0.063216 | 0.079648 |
| Concede | 0.002137 | 0.002149 | 0.014236 | 0.015540 |

The score model improves Brier score by 11.34% relative to the training-prevalence baseline; the conceding improvement is only 0.56%. There are just 175 positive scoring labels and 25 positive conceding labels in five test matches, and nearby action labels are correlated. These are preliminary probability results, not confidence intervals, causal action values, tactical ground truth or coaching validation. The model underpredicts average scoring probability in this test slice (0.00984 predicted versus 0.01507 observed). Do not infer a production-quality probability model from the average scores.

**Prior project exposure:** test match `3788742` (Denmark–Finland) had already been used as the evaluation match in the earlier two-match xT smoke test (`artifacts/integrations/statsbomb-spadl/source-manifest.json`). It was not used to select the new classifiers, and no splits or metrics were changed after the new test evaluation. The five-match test is held out from this experiment's fitting and model selection, but it is **not a pristine never-seen holdout across the whole project**. Future model comparisons need a new untouched corpus.

Actual corpus: `artifacts/datasets/statsbomb-euro2020-30/`. Actual model, test probabilities, test VAEP values, xT grid and per-file hashes: `artifacts/experiments/statsbomb-expanded-v1/`. Compact reviewable records are [dataset manifest](../../results/experiments/statsbomb-expanded-dataset.json) and [training report](../../results/experiments/statsbomb-expanded-training.json).

## Reproduce

Use the isolated analysis environment prepared by `scripts/setup/setup_integrations.py analysis`. The core application uses a different NumPy version. The tested runtime is socceraction 1.5.3, scikit-learn 1.9.0 and NumPy 1.26.4; exact versions are also in the experiment report.

```sh
cat > /tmp/statsbomb-corpus-request.json <<'JSON'
{"schema_version":1,"out":"artifacts/datasets/statsbomb-euro2020-30","commit":"4b73468fc5b0f1950f9f66fada70ad3a4f9327cb","competition_id":55,"season_id":43,"max_matches":30}
JSON
PYTHONPATH=src .venvs/analysis/bin/python -m soccerviz.datasets.statsbomb_dataset \
  --request /tmp/statsbomb-corpus-request.json --response /tmp/statsbomb-corpus-response.json

cat > /tmp/statsbomb-train-request.json <<'JSON'
{"schema_version":1,"dataset":"artifacts/datasets/statsbomb-euro2020-30","out":"artifacts/experiments/statsbomb-expanded-reproduction"}
JSON
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venvs/analysis/bin/python \
  -m soccerviz.modeling.action_training --request /tmp/statsbomb-train-request.json \
  --response /tmp/statsbomb-train-response.json
```

The dataset importer can resume downloads and rejects changes to a completed source manifest. Training refuses to overwrite an experiment directory. Use another directory for a fresh run. Do not repeatedly tune against this already-inspected test split; reserve a new untouched evaluation corpus for subsequent model selection decisions.

To replay the saved classifier on an untouched match's correctly oriented SPADL file, use the same worker with:

```json
{"schema_version":1,"operation":"predict","model_dir":"artifacts/experiments/statsbomb-expanded-v1","actions":"artifacts/datasets/statsbomb-euro2020-30/test.parquet","orientation":"attacking_left_to_right","out":"artifacts/experiments/replayed-test-probabilities.parquet"}
```

Inference verifies the model hash, checks its feature contract, and rejects games used for training or development/model selection. It does not construct future labels. It remains the caller's responsibility to supply verified SPADL orientation and the correct match identity when using an external provider.

Validation:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venvs/analysis/bin/pytest -q \
  tests/test_statsbomb_dataset.py tests/test_action_training.py
```

Tests exercise order-independent splitting, source pinning, content verification, period-boundary label isolation, separate held-out values, model coefficient replay and rejection of training/development inference games. Upstream modeling semantics: [socceraction VAEP documentation](https://socceraction.readthedocs.io/en/latest/documentation/valuing_actions/vaep.html).
