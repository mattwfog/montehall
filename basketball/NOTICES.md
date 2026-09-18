# Third-party notices — montehall CV pipeline

This pipeline was built from scratch (2026-07) with a license-clean
component menu (design ruling D7/D9; full audit 2026-07-07). All model
weights used in production inference are trained by us; no AGPL code or
weights are used anywhere in this tree.

## Libraries (code)

| Component | License | Where used |
|---|---|---|
| RF-DETR (`rfdetr`, Roboflow) | Apache-2.0 | `pipeline/detect.py` — detector architecture + COCO pretrained init for fine-tuning |
| segmentation-models-pytorch | MIT | `pipeline/court_seg.py` — DeepLabV3+ court-landmark segmenter architecture |
| PARSeq (`baudm/parseq` torch hub) | Apache-2.0 | `pipeline/jersey.py`, `training/train_parseq.py` — OCR architecture + pretrained init (weights fine-tuned on our synthetic + roster-verified crops) |
| torchreid (`deep-person-reid`, K. Zhou) | MIT | `pipeline/embed.py`, `training/train_reid*.py` — OSNet architecture only; weights trained from scratch by us |
| SigLIP2 (`google/siglip2-base-patch16-224`) | Apache-2.0 | `pipeline/embed.py` — team-appearance embeddings (pretrained weights as released) |
| transformers (Hugging Face) | Apache-2.0 | `pipeline/embed.py` — SigLIP2 loading |
| supervision (Roboflow) | MIT | GPU extra — detection utilities |
| PyAV | BSD-3-Clause | `pipeline/video.py` — decode/encode |
| WASB (ball detection) | MIT | `pipeline/ball_track.py` — production ball-supply channel (full-frame + tiled passes fused with detector rows; since 2026-08); `training/eval_wasb.py` — evaluation. Weights fine-tuned by us |
| numpy / pyarrow / pydantic / Pillow | BSD/Apache/MIT | core |

## Training data

| Dataset | License | Where used |
|---|---|---|
| TrackID3x3 (crops + ground truth) | CC BY 4.0 (data); Apache-2.0 (repo ground-truth tooling) | ReID training (`training/trackid_to_reid.py`). Attribution: TrackID3x3 dataset authors. |
| Roboflow Universe: `basketball-cv` | CC BY 4.0 (uploader-declared on the Universe page; the API returns no license field — re-verify before any redistribution) | detector v2 training mix (`training/fetch_roboflow.py`) |
| Roboflow Universe: `player-detection-2` | CC BY 4.0 (same uploader-declared caveat) | detector v2 training mix |
| Our own game footage + synthetic jersey crops (`training/synth_jersey.py`) | proprietary | detector/ReID/OCR self-training |

## Provenance notes

- PARSeq's released pretrained weights were trained on academic OCR
  datasets; we fine-tune away from that distribution on our own synthetic
  and roster-verified crops before production use.
- No published sports/ReID checkpoint is used at inference time — the
  2026-07-07 audit found all of them research-data-tainted; every
  production checkpoint in this pipeline is trained by us from the data
  listed above.
- The legacy pipeline's AGPL surface (ultralytics + YOLO
  weights) is not used, referenced, or ported here.
