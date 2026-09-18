"""Anonymous, short-range tracklet continuity; never equates a track ID with a name."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou(a, b):
    low, high = (
        np.maximum(a[:, None, :2], b[None, :, :2]),
        np.minimum(a[:, None, 2:], b[None, :, 2:]),
    )
    intersection = np.maximum(high - low, 0).prod(axis=2)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(axis=1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-6)


@dataclass
class Track:
    box: np.ndarray
    velocity: np.ndarray
    timestamp: float
    class_id: int
    appearance: np.ndarray | None = None


class TrackletAssociator:
    def __init__(self, max_age_s=0.8):
        self.max_age_s = max_age_s
        self.tracks = {}
        self.next_id = 1

    def reset(self):
        self.tracks.clear()  # IDs remain globally distinct across cuts.

    def update(self, boxes, classes, timestamp, appearances=None):
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        appearances = [None] * len(boxes) if appearances is None else appearances
        self.tracks = {
            k: v for k, v in self.tracks.items() if timestamp - v.timestamp <= self.max_age_s
        }
        result = np.full(len(boxes), -1, dtype=int)
        ids = list(self.tracks)
        if ids and len(boxes):
            predicted = np.stack(
                [
                    self.tracks[k].box
                    + self.tracks[k].velocity * (timestamp - self.tracks[k].timestamp)
                    for k in ids
                ]
            )
            overlap = box_iou(predicted, boxes)
            centers_a = (predicted[:, :2] + predicted[:, 2:]) / 2
            centers_b = (boxes[:, :2] + boxes[:, 2:]) / 2
            scale = np.linalg.norm(predicted[:, 2:] - predicted[:, :2], axis=1)
            distance = np.linalg.norm(centers_a[:, None] - centers_b[None], axis=2) / np.maximum(
                scale[:, None], 1
            )
            allowed = ((overlap > 0.05) | (distance < 0.7)) & (
                np.array([self.tracks[k].class_id for k in ids])[:, None]
                == np.asarray(classes)[None]
            )
            appearance_cost = np.zeros_like(distance)
            for i, key in enumerate(ids):
                old_color = self.tracks[key].appearance
                for j, color in enumerate(appearances):
                    if old_color is not None and color is not None:
                        difference = np.linalg.norm(old_color - color)
                        allowed[i, j] &= difference < 60
                        appearance_cost[i, j] = difference / 60
            cost = np.where(allowed, 1 - overlap + 0.3 * distance + appearance_cost, 1e6)
            ii, jj = linear_sum_assignment(cost)
            for i, j in zip(ii, jj, strict=True):
                if cost[i, j] < 1e6:
                    result[j] = ids[i]
        for j, box in enumerate(boxes):
            if result[j] < 0:
                result[j] = self.next_id
                self.next_id += 1
                velocity = np.zeros(4)
            else:
                old = self.tracks[result[j]]
                dt = max(timestamp - old.timestamp, 1e-3)
                velocity = 0.5 * old.velocity + 0.5 * (box - old.box) / dt
            color = appearances[j]
            previous = self.tracks.get(result[j])
            if previous is not None and previous.appearance is not None and color is not None:
                color = 0.8 * previous.appearance + 0.2 * color
            self.tracks[result[j]] = Track(box.copy(), velocity, timestamp, int(classes[j]), color)
        return result


def synthetic_identity_benchmark():
    rng = np.random.default_rng(22)
    tracker = TrackletAssociator()
    initial = rng.uniform([50, 50], [1800, 900], (22, 2))
    velocity = rng.normal(0, 10, (22, 2))
    identities, same, total = {}, 0, 0
    for frame in range(100):
        positions = initial + velocity * (frame / 5)
        order = rng.permutation(22)
        order = order[rng.random(22) > 0.08]
        centers = positions[order] + rng.normal(0, 1, (len(order), 2))
        boxes = np.column_stack([centers - [10, 25], centers + [10, 25]])
        assigned = tracker.update(boxes, np.zeros(len(order), int), frame / 5)
        for player, tid in zip(order, assigned, strict=True):
            if player in identities:
                same += int(identities[player] == tid)
                total += 1
            identities[player] = tid
    return {
        "benchmark": "synthetic 22-player image tracks, noise and 8% random missed detections",
        "identity_continuity_fraction": same / total,
        "associations": total,
        "real_video_identity_accuracy": "unmeasured without track annotations",
        "named_identity": "abstains: roster and jersey evidence unavailable",
    }
