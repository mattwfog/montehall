# Soccer CV and tactical-model repository survey

Research date: September 7, 2026. Requested window: April 7–September 7, 2026.
Mission: identify useful components for SoccerViz's elite tactical analysis and
managerial modeling, informed by the Montehall review.

## Findings

Roboflow sports is a useful visual starting point. The strongest new tactical
research leads are Monte Carlo Pass Search, GenTac, and TacticGen. Their release
states differ substantially: the first has implementation code but unresolved
checkpoint distribution; the latter two currently publish no model implementation.
The new offscreen-impute and junk-possession repositories provide smaller,
inspectable examples of reasoning about off-ball space.

Dates below come from GitHub repository metadata and default-branch commits, with
paper dates stated separately. Creation dates do not establish when a repository
became public. A recent `pushed_at` or search-engine crawl is not treated as a new
implementation release. The [API snapshot](../../results/research/github-snapshot-2026-09-07.json)
records file inventories and sampled commits; commit lists are capped, not full
histories. This was source inspection, not execution or reproduction of results.

## Roboflow sports: what we could use

[Repository](https://github.com/roboflow/sports) — created May 13, 2024. Latest
default-branch commit inspected:
[May 27, 2025, `42c80c0`](https://github.com/roboflow/sports/commit/42c80c06b6b65a7f89455b89fe31cdf4c38ba227).
The API reports an August 2026 push, but no default-branch commits in the requested
window. It is included by request, not because it is new.

The [soccer example](https://github.com/roboflow/sports/blob/42c80c06b6b65a7f89455b89fe31cdf4c38ba227/examples/soccer/main.py)
contains YOLO player/pitch/ball models, ByteTrack, team classification, and a
top-down radar. The ball demo uses tiled inference. The radar merges players,
goalkeepers, and referees; it does **not** run the separate ball detector. Its
displayed numbers are tracker IDs. The example does not implement named-player
identity, tactical learning, or persistent game-state export.

[Team classification](https://github.com/roboflow/sports/blob/42c80c06b6b65a7f89455b89fe31cdf4c38ba227/sports/common/team.py)
uses SigLIP crop embeddings, UMAP, and two-cluster KMeans. Goalkeeper affiliation
in the example is a nearest-team-centroid heuristic. These are useful baselines,
not established full-match identity solutions.

[Pitch projection](https://github.com/roboflow/sports/blob/42c80c06b6b65a7f89455b89fe31cdf4c38ba227/sports/common/view.py)
uses `cv2.findHomography`. The inspected implementation has no temporal calibration
filter or explicit robust-estimation method selection. The radar filters keypoints
by coordinates before fitting; we would need to measure projection stability on
our footage.

The [setup script](https://github.com/roboflow/sports/blob/42c80c06b6b65a7f89455b89fe31cdf4c38ba227/examples/soccer/setup.sh)
lists downloads for three weights and sample clips. Download availability and
inference were not tested. Repository code is MIT; that label does not establish
the terms of every dependency, dataset, or checkpoint.

**My assessment:** useful detector, pitch, and visualization experiments to feed
Montehall-style timestamped evidence. Adopting the entire demo as the tactical
model would leave most of the intended product unimplemented.

## New repositories inside the requested window

| Repository | Verified timing | Relevance | What is actually available |
| --- | --- | --- | --- |
| [Monte Carlo Pass Search](https://github.com/andrewkang12345/monteCarloPassSearch) | Created April 10; code import April 10; paper June 9 | Alternative passes, player rollouts, next touch, ball continuation, possession-value scoring | Python model/training/search code and small vocabulary assets. Checkpoint links are TODO. No root license found in inspected tree. |
| [GenTac](https://github.com/jyrao/GenTac) | Created April 9; paper April 13 | Conditional open-play trajectory generation and tactical-event forecasting | README, figures, and empty code/data placeholders. Code, TacBench, and weights marked coming soon. No license detected. |
| [TacticGen](https://github.com/Jasonxu1225/TacticGen) | Created April 21; paper April 20 | Joint player/ball tactical generation guided by rules, language, or learned objectives | Website and paper materials. Model code, weights, data, and documentation explicitly unreleased. Root license Apache-2.0. |
| [offscreen-impute](https://github.com/nowayfootball/offscreen-impute) | Created July 11; July paper; August 19 documentation update | How missing players distort pitch control and team control share | CPU benchmark, imputation policies, pitch-control implementation, and Metrica download script. MIT code. |
| [junk-possession](https://github.com/nowayfootball/junk-possession) | Created July 19; August paper; August 31 metadata updates | Distinguish low-threat possession that creates space from possession that fails to move the defensive block | Event index, spatial space-creation calculations, validation scripts. MIT code; underlying event corpus and broadcast pipeline are not included. |
| [EAST-SPL](https://github.com/AbolfazlChM95/EAST-SPL) | Created April 20; August 12 sample/dependency updates | Reduce computation in soccer player localization through adaptive tiling and empty-tile rejection | Four component subprojects, sample assets, and component instructions. MIT code; consolidated reproduction guide still pending. |

### The most relevant tactical research

**Monte Carlo Pass Search** is the closest released code to a bounded decision
model: generate alternative executions and options, roll out the next interaction,
and score the resulting state. It uses synchronized event/tracking data from a
public seven-match Bundesliga/2. Bundesliga dataset, including 3D ball information.
It is a tracking-data research pipeline rather than a broadcast-video extractor.
[Repository](https://github.com/andrewkang12345/monteCarloPassSearch),
[paper](https://arxiv.org/abs/2606.11120).

The paper says checkpoints were released, but the inspected
[checkpoint manifest](https://github.com/andrewkang12345/monteCarloPassSearch/blob/cc6af7b00769fc780d890eca49f530c2dce97e4b/CHECKPOINTS.md)
contains local machine paths and `TODO` download links. The
[runner](https://github.com/andrewkang12345/monteCarloPassSearch/blob/cc6af7b00769fc780d890eca49f530c2dce97e4b/monteCarloPassSearch/initParamVar/pass_mc_runner.py)
also defaults to author-local model/data paths. Treat it as code to study and
potentially reproduce, not an immediately usable pretrained model.

**GenTac and TacticGen** deserve reading for the longer-term model direction.
Both work with structured trajectories and context. GenTac connects sampled
futures to tactical events; TacticGen uses a diffusion transformer and objective
guidance. Their publication claims have not been independently reproduced here.
Neither repository currently supplies the implementation needed to test those
claims. [GenTac paper](https://arxiv.org/abs/2604.11786),
[TacticGen paper](https://arxiv.org/abs/2604.18210).

### Off-ball analysis we can inspect now

**offscreen-impute** implements last-seen, formation-offset, velocity, and centroid
voting baselines. Its public benchmark applies a simulated broadcast viewport to
three Metrica matches, with a held-out half from the third. It does not establish
robustness to noisy detections, wrong identities, or calibration errors from a
real broadcast extractor. Its useful contribution for us is testing the effect
of missing players on a downstream tactical measurement.
[Code and benchmark scope](https://github.com/nowayfootball/offscreen-impute).

**junk-possession** is especially close to the intended coaching questions. Its
spatial layer measures changes in control of advanced pitch zones. However, the
released scripts expect an event store and projected spatial series; the actual
broadcast extraction pipeline is absent. Its reported case studies are exploratory
evidence, not a validated general-purpose measure of managerial quality.
[Code and reproduction limitations](https://github.com/nowayfootball/junk-possession).

**EAST-SPL** is a more specialized efficiency lead for calibrated localization.
Its use of event/location statistics to allocate compute is relevant if full-pitch
processing becomes expensive. It does not supply a tactical reasoning model or
solve the moving broadcast-camera problem by itself.
[Implementation scope](https://github.com/AbolfazlChM95/EAST-SPL).

## Older repositories with meaningful updates in the window

| Repository | Recent change | Use for SoccerViz | Integration facts |
| --- | --- | --- | --- |
| [FOOTPASS](https://github.com/JeremieOchin/FOOTPASS) | April 16 token filtering; May 7 training changes; June citation updates | Learn and evaluate who performs which ball action and when, with graph and sequence-context baselines | Training/inference/evaluation code exists. Dataset describes 54 full matches; video access needs SoccerNet's NDA. |
| [SoccerNet sn-gamestate](https://github.com/SoccerNet/sn-gamestate) + [TrackLab](https://github.com/TrackingLaboratory/tracklab) | April 30–May 2 compatibility fixes; TrackLab 1.3.24 repairs GS-HOTA evaluation | A shared benchmark for positions, team, role, jersey, and temporal identity | Existing GSR implementation and evaluation framework. sn-gamestate root GPL-3.0; TrackLab root MIT. |
| [Broadcast2Pitch / SoccernetGSR](https://github.com/yinmayoo185/SoccernetGSR) | June 2 robust line/circle fitting and OpenCV dependency fix | Calibration, tracking, ReID, role/jersey recognition, and tracklet refinement | Implementation exists; README specifies Linux/NVIDIA/CUDA. No root license found in inspected tree. |

FOOTPASS has a material documentation conflict: the
[root LICENSE](https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/LICENSE)
is Apache-2.0, while the
[README](https://github.com/JeremieOchin/FOOTPASS/blob/f63f37c1a0c6bcebf568a8ce716495a459376af5/README.md)
states CC BY-NC 4.0 for annotations and baselines. Commercial reuse terms need
clarification. Its README clone URL points to `Footovision/FOOTPASS`, which returned
404; `JeremieOchin/FOOTPASS` is the accessible repository inspected here.

FOOTPASS's sequence transducer consumes predictions from its visual model. That
differs from Montehall's primitive-only brain contract. Its supervision and
evaluation remain relevant without copying that architectural choice.

## Relevant but not new in this window

- [SPL-BEV](https://github.com/IvarPersson/SPL-BEV): created January 2025;
  latest reported push February 19, 2026; no commits in the requested window.
  Code and `.pt` model files exist despite the README's stale coming-soon sentence.
- [ExpectedPassTurnovers](https://github.com/andypetes94/ExpectedPassTurnovers):
  November 2025 repository with no commits in the window. Relevant to pressing
  targets, but a 2026 application paper does not make the implementation new.
- TacticAI and the original Metrica sample data remain useful background, already
  linked in the [Montehall review](montehall-review.md).

## Suggested order of investigation

My recommendation, not an implementation commitment:

1. Use Roboflow's example for a small visual baseline, and assess reconstruction
   with the SoccerNet/TrackLab evaluation concepts. Preserve timestamps and raw
   evidence using the Montehall lessons.
2. Inspect offscreen-impute's public benchmark to establish how incomplete
   visibility changes the tactical conclusions we intend to show.
3. Study FOOTPASS for player-action supervision; resolve data access and the
   conflicting terms before committing it to a commercial training path.
4. Prototype a bounded tactical question over reliable tracking: space creation
   or pass alternatives. junk-possession and Monte Carlo Pass Search provide
   concrete reference code, with the release limitations above.
5. Keep GenTac and TacticGen as research references and release-watch candidates.
   Their availability should not block our first measured experiments.

No packages, model weights, datasets, or remote compute were installed or started.
Repository presence and file inspection establish available materials, not
successful execution, scientific validity, or elite coaching performance.
