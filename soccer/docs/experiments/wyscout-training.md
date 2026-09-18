# Wyscout open-data training and evaluation (CC BY 4.0)

The permissive replacement for the [StatsBomb baseline](statsbomb-training.md),
whose agreement forbids commercial exploitation of the data or any analysis
derived from it. Source: the Wyscout open match-event dataset (Pappalardo et al.
2019, figshare collection 4415000), every article read as **CC BY 4.0** on
2026-09-08. `soccerviz.datasets.wyscout_dataset` downloads the nine files by
figshare id, verifies the published MD5 and records SHA-256 and byte counts in
`artifacts/public-data/wyscout/source.json`. Attribution: **Data: Wyscout via
Pappalardo et al. 2019, CC BY 4.0**.

## Corpus

Seven competitions of one season: Serie A, Premier League, La Liga, Ligue 1 and
Bundesliga 2017/18, Euro 2016 and World Cup 2018, 1,941 games in total.
socceraction 1.5.3's own public Wyscout loader and SPADL converter are used
(the extracted files are linked into the flat layout it expects), then the
same left-to-right orientation and validation as the StatsBomb corpus. The
deterministic split (`sha256("soccerviz-v1:" + match_id)`) is written to
`splits.json` before any event is converted, over the pooled game list; teams
recur across splits.

| Partition | Matches | SPADL actions |
|---|---:|---:|
| Train | 200 | 253,681 |
| Development | 50 | 63,567 |
| Test | 50 | 63,729 |

Corpus: `artifacts/datasets/wyscout-7comp-300/`; compact record
[dataset manifest](../../results/experiments/wyscout-7comp-300-dataset.json).

## What was trained

Exactly the StatsBomb protocol, unchanged: socceraction xT grid fitted on the
training matches, and the two standardised logistic VAEP classifiers (scoring,
conceding) over 44 current-action features, `C` in {0.1, 1.0} selected on the
development log loss (0.1 both). Model, test probabilities, VAEP values and
xT grid: `artifacts/experiments/wyscout-7comp-300-v1/`; report
[training report](../../results/experiments/wyscout-7comp-300-training.json).

## Held-out results, 50 test matches

| Target | Model Brier ↓ | Train-prevalence Brier ↓ | Model log loss ↓ | Prevalence log loss ↓ | Positives |
|---|---:|---:|---:|---:|---:|
| Score | 0.013737 | 0.015521 | 0.071265 | 0.081088 | 1,005 of 63,729 |
| Concede | 0.005018 | 0.005074 | 0.029975 | 0.032006 | 325 of 63,729 |

Brier skill against the training prevalence: 11.5% for scoring (StatsBomb
Euro 2020 on 5 test matches: 11.3%) and 1.1% for conceding (StatsBomb 0.6%).
The scoring model is calibrated on average here (mean predicted 0.01542
versus observed 0.01577), where the StatsBomb model underpredicted (0.00984
versus 0.01507). The test slice has six times the positives of the StatsBomb
one, so these numbers are the more stable of the two, but they remain a
modest current-action baseline, not the full VAEP architecture and not a
coaching-quality value model.

## Consequences

- The action-value models in the tree are now trainable and shippable on
  permissive data. The StatsBomb corpus and its models stay as research
  artefacts; nothing consumes them for a product.
- The conceding head barely beats prevalence on either corpus; it needs the
  full three-action feature set before it means anything.

## Reproduce

```sh
PYTHONPATH=src .venvs/analysis/bin/python -m soccerviz.datasets.wyscout_dataset \
  --spadl-out artifacts/datasets/wyscout-7comp-300 --max-matches 300
PYTHONPATH=src .venvs/analysis/bin/python -m soccerviz.modeling.action_training \
  --request artifacts/experiments/wyscout-7comp-300-train-request.json \
  --response results/experiments/wyscout-7comp-300-training.json
```
