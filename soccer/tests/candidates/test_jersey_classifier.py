"""Source parsing, broadcast-scale degradation, the per-crop decision rule and the inference contract."""

import json
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest

from soccerviz.candidates import jersey_classifier as jc
from soccerviz.core.assets import sha256

TAISEIS = Path("artifacts/public-data/roboflow/jersey-number-ijbaq-v1-folder")
PUSAN = Path("artifacts/public-data/roboflow/jersey-number-detection-8a55j-v1-coco")


def test_folder_records_map_abstain_classes_and_drop_double_zero():
    if not TAISEIS.exists():
        pytest.skip("taiseis export not downloaded")
    records = jc.folder_records(TAISEIS, "valid")
    labels = {r["label"] for r in records}
    assert jc.ILLEGIBLE in labels and max(l for l in labels if l != jc.ILLEGIBLE) < 100
    assert not any("/00/" in r["image_path"] for r in records)
    assert all(Path(r["image_path"]).exists() for r in records[:5])


def test_pusan_records_read_numbers_left_to_right_and_make_negatives():
    if not PUSAN.exists():
        pytest.skip("Pusan export not downloaded")
    records = jc.pusan_records(PUSAN, "valid")
    numbers = [r for r in records if r["source"] == "pusan"]
    negatives = [r for r in records if r["source"] == "pusan-negative"]
    assert numbers and negatives and all(r["label"] == jc.ILLEGIBLE for r in negatives)
    assert all(0 <= r["label"] < 100 for r in numbers)
    crop = jc.load_crop(numbers[0])
    assert crop.ndim == 3 and crop.shape[0] > 8 and crop.shape[1] > 8


def test_degradation_reaches_broadcast_torso_heights():
    rng = np.random.default_rng(3)
    crop = np.full((224, 224, 3), 200, dtype=np.uint8)
    heights = {jc.degrade(crop, rng).shape[0] for _ in range(50)}
    assert min(heights) >= jc.TORSO_HEIGHT_RANGE[0] and max(heights) <= jc.TORSO_HEIGHT_RANGE[1]
    tensor = jc.preprocess(jc.jitter_crop(jc.degrade(crop, rng), rng))
    assert tensor.shape == (3, jc.INPUT_SIZE, jc.INPUT_SIZE) and tensor.dtype == np.float32


def test_read_distribution_abstains_on_low_legibility_or_low_probability():
    p = np.zeros(jc.NUM_CLASSES)
    p[7], p[jc.ILLEGIBLE] = 0.3, 0.6
    assert jc.read_distribution(p)["status"] == "abstain"
    p[7], p[jc.ILLEGIBLE] = 0.85, 0.05
    out = jc.read_distribution(p)
    assert out["status"] == "provisional" and out["jersey_number"] == 7
    assert out["legibility"] == pytest.approx(0.95)


class FakeClassifier:
    """Returns a fixed distribution per crop so the manifest contract can be checked without torch."""

    metadata: ClassVar[dict] = {"name": "fake"}

    def __init__(self, distributions):
        self.distributions_by_index = distributions

    def distributions(self, crops):
        return np.stack(self.distributions_by_index[: len(crops)])


def test_infer_writes_predictions_and_distributions_bound_to_the_manifest(tmp_path):
    crops_dir = tmp_path / "crops"
    crops_dir.mkdir()
    rows = []
    for i in range(3):
        path = crops_dir / f"c{i}.png"
        cv2.imwrite(str(path), np.full((90, 40, 3), 60 + i, dtype=np.uint8))
        rows.append(
            {
                "crop_id": f"c{i}",
                "crop_path": f"crops/c{i}.png",
                "crop_sha256": sha256(path),
                "torso_bounds_in_crop_xyxy": [0, 15, 40, 60],
            }
        )
    manifest = tmp_path / "crop-manifest.json"
    manifest.write_text(json.dumps({"schema": "jersey-crops/v1", "crops": rows}))
    d = np.zeros((3, jc.NUM_CLASSES))
    d[0, 10], d[0, jc.ILLEGIBLE] = 0.9, 0.05
    d[1, jc.ILLEGIBLE] = 0.9
    d[2, 23], d[2, 28] = 0.45, 0.45
    out = jc.infer(manifest, "unused.pt", tmp_path / "out", model=FakeClassifier(list(d)))
    statuses = [p["status"] for p in out["predictions"]]
    assert statuses == ["provisional", "abstain", "abstain"]
    assert out["predictions"][0]["jersey_number"] == 10
    assert out["crop_manifest_sha256"] == sha256(manifest) and out["complete_crop_manifest"]
    stored = np.load(tmp_path / "out" / "distributions.npz")
    assert stored["crop_ids"].tolist() == ["c0", "c1", "c2"]
    assert np.allclose(stored["probabilities"], d)


FONTS = Path("artifacts/public-data/fonts")
POOL = Path("artifacts/integrations/jersey-torso-pool")


def test_render_number_changes_only_the_torso_it_draws_on():
    if not FONTS.exists():
        pytest.skip("fonts not fetched")
    fonts = jc.load_fonts(FONTS)
    rng = np.random.default_rng(0)
    torso = np.full((48, 40, 3), 40, dtype=np.uint8)  # a dark kit
    rendered = jc.render_number(torso, 23, fonts, rng)
    assert rendered.shape[0] == jc.RENDER_HEIGHT and rendered.ndim == 3
    plain = cv2.resize(torso, (rendered.shape[1], rendered.shape[0]), interpolation=cv2.INTER_CUBIC)
    changed = (np.abs(rendered.astype(int) - plain.astype(int)).sum(axis=2) > 30).mean()
    assert 0.02 < changed < 0.6  # digits cover part of the torso, not all of it
    assert rendered[:, :, :].mean() > plain.mean()  # light digits on a dark kit


def test_detection_torso_records_follow_the_torso_rule():
    export = Path("artifacts/public-data/roboflow/football-players-detection-3zvbc-v10-coco")
    if not export.exists():
        pytest.skip("v10 export not downloaded")
    records = jc.detection_torso_records(export, "valid")
    assert len(records) > 500 and all(r["label"] == jc.ILLEGIBLE for r in records)
    x0, y0, x1, y1 = records[0]["region"]
    assert x1 > x0 and y1 > y0
    if POOL.exists():
        assert len(list((POOL / "valid").glob("*.png"))) == len(records)


def test_temperature_is_applied_at_inference(tmp_path):
    torch = pytest.importorskip("torch")
    net = jc.build_network(pretrained=False)
    for temperature in (1.0, 2.5):
        torch.save({"model": net.state_dict(), "temperature": temperature}, tmp_path / "m.pt")
        model = jc.JerseyClassifier(tmp_path / "m.pt", device="cpu", batch_size=2)
        assert model.temperature == temperature
    crops = [np.full((44, 40, 3), 90, dtype=np.uint8)]
    sharp = jc.JerseyClassifier.__new__(jc.JerseyClassifier)
    sharp.torch, sharp.device, sharp.batch_size, sharp.net, sharp.temperature = (
        torch,
        "cpu",
        2,
        model.net,
        1.0,
    )
    soft = jc.JerseyClassifier.__new__(jc.JerseyClassifier)
    soft.torch, soft.device, soft.batch_size, soft.net, soft.temperature = (
        torch,
        "cpu",
        2,
        model.net,
        2.5,
    )
    assert soft.distributions(crops).max() < sharp.distributions(crops).max()
