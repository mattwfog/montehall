import copy
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from soccerviz.candidates.detector_training import (
    CLASSES,
    DATASET_REVISION,
    FRAME_IDS,
    RF_TRAINING,
    clipped_xywh,
    explicit_ignore,
    export_corpus,
    training_annotations,
    validate_partition,
    verify_corpus,
    write_new_json,
    yolo_line,
)
from soccerviz.core.assets import sha256


def protocol():
    return {
        "source_split": "train",
        "revision": DATASET_REVISION,
        "excluded_external_game_ids": ["2", "3", "5"],
        "training": copy.deepcopy(RF_TRAINING),
        "sequences": [
            {
                "sequence": f"SNGS-{i:03d}",
                "game_id": str(game),
                "partition": "train" if i < 3 else "valid",
                "frame_ids": list(FRAME_IDS),
            }
            for i, game in enumerate((4, 6, 9), 1)
        ],
    }


def annotation(identifier, category, box, **extra):
    return {
        "id": identifier,
        "category_id": category,
        "supercategory": "object",
        "bbox_image": dict(zip(("x", "y", "w", "h"), box, strict=True)),
        **extra,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(source_split="valid"),
        lambda p: p["sequences"][0].update(game_id="2"),
        lambda p: p["sequences"][2].update(game_id="4"),
        lambda p: p["training"].update(epochs=6),
        lambda p: p["sequences"][0].update(frame_ids=[1, 2]),
    ],
)
def test_training_protocol_rejects_leakage_or_changed_budget(mutation):
    p = protocol()
    validate_partition(p)
    mutation(p)
    with pytest.raises(ValueError):
        validate_partition(p)


def test_difficult_boolean_strings_do_not_turn_false_into_ignore():
    assert not explicit_ignore({"attributes": {"difficult": "false", "ignore": "0"}})
    assert explicit_ignore({"attributes": {"difficult": "true"}})
    assert explicit_ignore({"iscrowd": 1})


def test_ignore_and_other_regions_are_masked_in_both_export_formats():
    image = {
        "width": 20,
        "height": 10,
        "ignore_regions_x": [[0, 4, 4, 0]],
        "ignore_regions_y": [[0, 0, 4, 4]],
    }
    annotations = [
        annotation("kept", 1, [10, 1, 2, 4]),
        annotation("region", 1, [1, 1, 2, 2]),
        annotation("other", 7, [15, 1, 2, 2]),
        annotation("difficult", 4, [7, 7, 2, 2], difficult=True),
    ]
    retained, mask, metadata = training_annotations(
        image, annotations, {1: "player", 4: "ball", 7: "other"}
    )
    assert [r["source_annotation_id"] for r in retained] == ["kept"]
    assert mask[1, 15] == mask[7, 7] == mask[2, 2] == 1
    assert metadata["masked_pixels"] > 0
    assert {r["reason"] for r in metadata["dropped_annotations"]} == {
        "other_category",
        "explicit_ignore",
        "majority_in_ignore_region",
    }


def test_boxes_clip_to_source_pixels_and_yolo_round_trips():
    box = clipped_xywh(annotation("a", 1, [-2, 1, 7, 5]), 20, 10)
    assert box == [0, 1, 5, 5]
    values = yolo_line({"class_id": 0, "bbox_xywh": box}, 20, 10).split()
    assert values[0] == "0"
    cx, cy, w, h = map(float, values[1:])
    assert [(cx - w / 2) * 20, (cy - h / 2) * 10, w * 20, h * 10] == pytest.approx(box)
    with pytest.raises(ValueError, match="Invalid source box"):
        clipped_xywh(annotation("bad", 1, [0, 0, np.nan, 4]), 20, 10)


def test_shared_exports_preserve_class_order_and_detect_modified_images(tmp_path):
    p = protocol()
    downloaded = []
    for sequence in p["sequences"]:
        images, annotations = [], []
        for fid in FRAME_IDS:
            name = f"{fid:06d}.jpg"
            path = tmp_path / "sources" / "train" / sequence["sequence"] / "img1" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.full((10, 20, 3), fid % 255, dtype=np.uint8))
            images.append({"image_id": fid, "file_name": name, "width": 20, "height": 10})
            for class_id in range(1, 5):
                annotations.append(
                    annotation(f"{fid}_{class_id}", class_id, [class_id * 3, 1, 2, 5], image_id=fid)
                )
            downloaded.append(
                {"relative_path": str(path.relative_to(tmp_path)), "sha256": sha256(path)}
            )
        source = {
            "info": {"version": "1.3"},
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)],
        }
        label_path = path.parent.parent / "Labels-GameState.json"
        write_new_json(label_path, source)
        sequence["annotation_path"] = str(label_path.relative_to(tmp_path))
        sequence["annotation_sha256"] = sha256(label_path)
    write_new_json(tmp_path / "protocol.json", p)
    write_new_json(tmp_path / "download-summary.json", {"files": downloaded})
    result = export_corpus(tmp_path)
    assert result["frame_counts"] == {"train": 150, "valid": 75}
    assert result["census"]["train"] == dict.fromkeys(CLASSES, 150)
    coco = json.loads((tmp_path / "coco/train/_annotations.coco.json").read_text())
    assert [row["name"] for row in coco["categories"]] == list(CLASSES)
    first = result["frames"][0]
    image_path = tmp_path / first["image_path"]
    yolo_path = tmp_path / "yolo/images/train" / image_path.name
    assert sha256(image_path) == sha256(yolo_path)
    lines = (
        (tmp_path / "yolo/labels/train" / image_path.with_suffix(".txt").name)
        .read_text()
        .splitlines()
    )
    assert [line.split()[0] for line in lines] == ["0", "1", "2", "3"]
    verify_corpus(tmp_path)
    image_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="training file changed"):
        verify_corpus(tmp_path)
    with pytest.raises(FileExistsError):
        export_corpus(tmp_path)


def test_training_runner_refuses_to_silently_remap_nonzero_gpu():
    spec = importlib.util.spec_from_file_location(
        "train_candidate",
        Path(__file__).resolve().parents[2] / "scripts/training/train_detector_candidate.py",
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    base = ["--checkpoint", "weights.pth", "--checkpoint-sha256", "0" * 64, "--out", "unused"]
    assert runner.parse_args(base + ["--device", "cuda:0"]).device == "cuda:0"
    with pytest.raises(SystemExit):
        runner.parse_args(base + ["--device", "cuda:1"])


def test_setup_failure_is_recorded_without_claiming_training(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "train_candidate_failure",
        Path(__file__).resolve().parents[2] / "scripts/training/train_detector_candidate.py",
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    output = tmp_path / "failed-smoke"

    def fail_setup(args):
        args.out.mkdir()
        args.created_output = True
        args.execution_phase = "setup"
        raise AttributeError("fixture upstream setup failure")

    monkeypatch.setattr(runner, "train", fail_setup)
    with pytest.raises(AttributeError, match="upstream setup failure"):
        runner.main(
            [
                "--checkpoint",
                "unused.pth",
                "--checkpoint-sha256",
                "0" * 64,
                "--out",
                str(output),
                "--smoke-step",
            ]
        )
    report = json.loads((output / "failure.json").read_text())
    assert report["completed"] is False
    assert report["optimizer_update_verified"] is False
    assert report["phase"] == "setup"
    assert not (output / "training-evidence.json").exists()
