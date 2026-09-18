"""The Roboflow field-keypoint index order is the vertex order, whatever the export's names say."""

import json
from pathlib import Path

import numpy as np
import pytest

from soccerviz.core.geometry import calibrate, pitch_landmarks

EXPORT = Path("artifacts/public-data/roboflow/football-field-detection-f07vi-v12-coco/train")


def test_landmark_table_is_the_roboflow_sports_vertex_order():
    table = pitch_landmarks()
    assert table.shape == (32, 2)
    assert tuple(table[0]) == (0.0, 0.0)
    assert tuple(table[13]) == (52.5, 0.0)  # vertex 14, halfway line touchline
    assert tuple(table[14]) == (52.5, 34.0 - 9.15)  # vertex 15, centre-circle top
    assert tuple(table[30]) == (52.5 - 9.15, 34.0)  # vertex 31, centre-circle left
    assert tuple(table[29]) == (105.0, 68.0)  # vertex 30, far corner


def test_roboflow_export_keypoint_index_is_homography_consistent_with_the_table():
    annotations = EXPORT / "_annotations.coco.json"
    if not annotations.exists():
        pytest.skip("Roboflow v12 export not downloaded")
    data = json.loads(annotations.read_text())
    names = next(c for c in data["categories"] if c.get("keypoints"))["keypoints"]
    assert names[30:] == ["14", "19"], "the trap this test documents: names are not indices"
    by_name = pitch_landmarks()[[int(name) - 1 for name in names]]
    accepted = {"index": 0, "name": 0}
    for annotation in data["annotations"]:
        if annotation["category_id"] != 1:
            continue
        points = np.asarray(annotation["keypoints"], dtype=float).reshape(32, 3)
        visible = points[:, 2] > 0
        if visible.sum() < 6:
            continue
        accepted["index"] += calibrate(points[visible, :2], pitch_landmarks()[visible]).accepted
        accepted["name"] += calibrate(points[visible, :2], by_name[visible]).accepted
    assert accepted["index"] > 200 and accepted["name"] < 100, accepted
