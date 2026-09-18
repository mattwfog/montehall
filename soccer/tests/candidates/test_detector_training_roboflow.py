import copy
import json

import cv2
import numpy as np
import pytest

from soccerviz.candidates.detector_training import (
    CLASSES,
    ROBOFLOW_HOLDOUT_FRACTION,
    export_roboflow_corpus,
    freeze_roboflow_protocol,
    roboflow_partition,
    validate_roboflow_partition,
    verify_corpus,
    write_new_json,
)
from soccerviz.core.assets import sha256

VIDEOS = {"08fd33": 6, "42ba34": 4, "cd987c": 2}
ROBOFLOW_CATEGORIES = [
    {"id": 0, "name": "football-players-detection", "supercategory": "none"},
    {"id": 1, "name": "ball", "supercategory": "football-players-detection"},
    {"id": 2, "name": "goalkeeper", "supercategory": "football-players-detection"},
    {"id": 3, "name": "player", "supercategory": "football-players-detection"},
    {"id": 4, "name": "referee", "supercategory": "football-players-detection"},
]


def fake_export(source_dir):
    """A Roboflow-style export whose random splits leak every video across train/valid."""
    counter = 0
    splits = {"train": [], "valid": []}
    for video, count in VIDEOS.items():
        for k in range(count):
            counter += 1
            split = "valid" if k == 0 else "train"
            name = f"{video}_{k}_{k}_png.rf.{counter:032x}.jpg"
            path = source_dir / split / name
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.full((10, 20, 3), counter % 255, dtype=np.uint8))
            splits[split].append((counter, name))
    for split, rows in splits.items():
        images = [{"id": i, "file_name": n, "width": 20, "height": 10} for i, n in rows]
        annotations = [
            {
                "id": i * 10 + c,
                "image_id": i,
                "category_id": c,
                "bbox": [c * 3.0, 1.0, 2.0, 5.0],
                "area": 10.0,
                "iscrowd": 0,
            }
            for i, _ in rows
            for c in range(1, 5)
        ]
        write_new_json(
            source_dir / split / "_annotations.coco.json",
            {"images": images, "annotations": annotations, "categories": ROBOFLOW_CATEGORIES},
        )
    write_new_json(
        source_dir / "source.json",
        {
            "project_id": "roboflow-jvuqo/football-players-detection-3zvbc",
            "version_id": "roboflow-jvuqo/football-players-detection-3zvbc/10",
            "license": "CC BY 4.0",
            "archive_sha256": "0" * 64,
            "preprocessing": {"auto-orient": {"enabled": True}},
        },
    )


def external_manifest(tmp_path):
    path = tmp_path / "external.json"
    write_new_json(path, {"schema": "vision-benchmark/v1", "split": "valid", "frames": []})
    return path


def test_partition_is_video_disjoint_and_deterministic():
    images = {
        f"{v}_{k}_{k}": {"video": v} for v, count in VIDEOS.items() for k in range(count)
    }
    partition = roboflow_partition(images)
    assert partition == {"cd987c": "valid", "42ba34": "train", "08fd33": "train"}
    assert roboflow_partition(dict(reversed(images.items()))) == partition


def test_freeze_export_verify_round_trip(tmp_path):
    source = tmp_path / "source"
    fake_export(source)
    root = tmp_path / "corpus"
    root.mkdir()
    protocol = freeze_roboflow_protocol(root, source, external_manifest(tmp_path))
    assert protocol["videos"]["cd987c"]["partition"] == "valid"
    assert len(protocol["images"]) == sum(VIDEOS.values())
    manifest = export_roboflow_corpus(root)
    assert manifest["frame_counts"] == {"train": 10, "valid": 2}
    assert manifest["video_counts"] == {"train": 2, "valid": 1}
    assert manifest["census"]["train"] == dict.fromkeys(CLASSES, 10)
    coco = json.loads((root / "coco/train/_annotations.coco.json").read_text())
    assert [row["name"] for row in coco["categories"]] == list(CLASSES)
    # Roboflow id 3 ("player") lands on fixed id 1; ball (Roboflow 1) lands on 4.
    by_box = {tuple(a["bbox"]): a["category_id"] for a in coco["annotations"] if a["image_id"] == 1}
    assert by_box[(3.0, 1.0, 2.0, 5.0)] == 4 and by_box[(9.0, 1.0, 2.0, 5.0)] == 1
    first = manifest["frames"][0]
    label = (root / "yolo/labels" / first["partition"] / f"{first['frame_id']}.txt").read_text()
    assert sorted(line.split()[0] for line in label.splitlines()) == ["0", "1", "2", "3"]
    verified, _ = verify_corpus(root)
    assert verified["protocol_sha256"] == sha256(root / "protocol.json")
    (root / first["image_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="training file changed"):
        verify_corpus(root)
    with pytest.raises(FileExistsError):
        export_roboflow_corpus(root)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["training"].update(epochs=6),
        lambda p: p["videos"]["cd987c"].update(partition="train"),
        lambda p: p["videos"]["08fd33"]["images"].append(p["videos"]["42ba34"]["images"][0]),
        lambda p: p["videos"]["42ba34"]["images"].pop(),
    ],
)
def test_protocol_rejects_leakage_or_changed_budget(tmp_path, mutation):
    source = tmp_path / "source"
    fake_export(source)
    root = tmp_path / "corpus"
    root.mkdir()
    protocol = copy.deepcopy(freeze_roboflow_protocol(root, source, external_manifest(tmp_path)))
    mutation(protocol)
    with pytest.raises(ValueError):
        validate_roboflow_partition(protocol)


def test_holdout_fraction_is_enforced():
    assert 0 < ROBOFLOW_HOLDOUT_FRACTION < 0.5
