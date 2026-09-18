# Third-party notices

The code in this repository is licensed under Apache-2.0. Each project keeps its
own record of third-party components:

- Basketball: [`basketball/NOTICES.md`](basketball/NOTICES.md) lists libraries,
  pretrained weights and training data. No AGPL code or weights are used in
  `basketball/`.
- Soccer: [`soccer/docs/guides/public-data.md`](soccer/docs/guides/public-data.md)
  carries the dataset and model licence census, and
  `soccer/src/soccerviz/vision/checkpoint_licence.py` enforces it at load time.

## Optional copyleft dependency

`soccer/` declares `ultralytics` (AGPL-3.0) only in its optional `vision` extra.
It is imported lazily by the YOLO-based video pipeline, the YOLO detector and
calibration backends and the YOLO training script, and it is not installed by the
default `uv sync`. The RF-DETR detector backend is the permissive alternative.
Anyone who installs the `vision` extra
and distributes or serves the result takes on the AGPL-3.0 terms for that
combination.

Datasets, video and model weights are not redistributed here. Each is subject to
its own licence, recorded in the documents above.
