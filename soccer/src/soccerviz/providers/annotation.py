"""Offline CVAT XML review exchange; predictions never become implicit ground truth."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from soccerviz.core.assets import sha256
from soccerviz.harness.engine import digest
from soccerviz.vision.specialists import validate_input, validate_review

BOX_KEYS = ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")
ATTRIBUTES = {
    "reviewed": ("checkbox", "false"),
    "detection_id": ("text", ""),
    "identity": ("text", ""),
    "team": ("select", "unknown\n0\n1"),
}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export_cvat(evidence: dict, output: Path) -> dict:
    """Export annotations for the ORIGINAL video with zero-based source-frame indices."""
    validate_input(evidence)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("annotations.xml", "manifest.json")):
        raise ValueError("Use a new export directory to preserve existing review artifacts")
    frames = evidence["tables"]["frames"]
    frame_map = {f["frame_id"]: f for f in frames}
    if len({f["source_frame"] for f in frames}) != len(frames):
        raise ValueError("Source frame indices must be unique")
    size = max(f["source_frame"] for f in frames) + 2
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    task = ET.SubElement(ET.SubElement(root, "meta"), "task")
    for key, value in {
        "id": 0,
        "name": "SoccerViz human review",
        "size": size,
        "mode": "interpolation",
        "overlap": 0,
        "flipped": "False",
        "start_frame": 0,
        "stop_frame": size - 1,
        "frame_filter": "",
    }.items():
        ET.SubElement(task, key).text = str(value)
    original_size = ET.SubElement(task, "original_size")
    for key in ("width", "height"):
        ET.SubElement(original_size, key).text = str(frames[0][key])
    labels = ET.SubElement(task, "labels")
    roles = sorted({d["role_hypothesis"] for d in evidence["tables"]["detections"]})
    for role in roles:
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = role
        attrs = ET.SubElement(label, "attributes")
        for name, (kind, values) in ATTRIBUTES.items():
            attr = ET.SubElement(attrs, "attribute")
            for key, value in {
                "name": name,
                "mutable": "True",
                "input_type": kind,
                "default_value": values.split("\n")[0],
                "values": values,
            }.items():
                ET.SubElement(attr, key).text = value
    groups = {}
    for row in evidence["tables"]["detections"]:
        groups.setdefault(
            (row.get("shot_id", 0), row["tracklet_id"], row["role_hypothesis"]), []
        ).append(row)
    for tid, (key, rows) in enumerate(sorted(groups.items())):
        track = ET.SubElement(root, "track", id=str(tid), label=key[2], source="auto")
        rows.sort(key=lambda row: frame_map[row["frame_id"]]["source_frame"])
        for index, row in enumerate(rows):
            source_frame = frame_map[row["frame_id"]]["source_frame"]
            attrs = dict(zip(("xtl", "ytl", "xbr", "ybr"), map(str, (row[k] for k in BOX_KEYS))))
            box = ET.SubElement(
                track,
                "box",
                frame=str(source_frame),
                outside="0",
                occluded="0",
                keyframe="1",
                **attrs,
            )
            for name, value in {
                "reviewed": "false",
                "detection_id": row["detection_id"],
                "identity": "",
                "team": "unknown",
            }.items():
                ET.SubElement(box, "attribute", name=name).text = value
            next_frame = (
                frame_map[rows[index + 1]["frame_id"]]["source_frame"]
                if index + 1 < len(rows)
                else size
            )
            if source_frame + 1 < next_frame:
                ET.SubElement(
                    track,
                    "box",
                    frame=str(source_frame + 1),
                    outside="1",
                    occluded="0",
                    keyframe="1",
                    **attrs,
                )
    ET.indent(root)
    xml = output / "annotations.xml"
    ET.ElementTree(root).write(xml, encoding="utf-8", xml_declaration=True)
    manifest = {
        "schema": "cvat-source-manifest/v1",
        "input_id": digest(evidence),
        "source_sha256": evidence["source_sha256"],
        "model_sha256": evidence["model_sha256"],
        "annotations_sha256": sha256(xml),
        "coordinate_space": "original video pixels",
        "cvat_frame_index": "zero-based source_frame; create task from original video at step 1",
        "frames": [
            {
                k: f[k]
                for k in (
                    "frame_id",
                    "source_frame",
                    "timestamp_s",
                    "pts",
                    "time_base",
                    "width",
                    "height",
                )
            }
            for f in frames
        ],
        "labels": roles,
        "review_status": "unreviewed",
    }
    _write_json(output / "manifest.json", manifest)
    template = {
        "schema": "cvat-review/v1",
        "manifest_id": digest(manifest),
        "annotations_sha256": "",
        "reviewer": "",
        "reason": "",
        "reviewed_frames": [],
    }
    _write_json(output / "review-template.json", template)
    report = {
        "schema": "cvat-export-report/v1",
        "status": "awaiting_human_review",
        "frames": len(frames),
        "suggested_boxes": len(evidence["tables"]["detections"]),
        "reviewed_boxes": 0,
        "hota": None,
        "idf1": None,
        "annotations": str(xml.resolve()),
        "manifest_id": digest(manifest),
    }
    _write_json(output / "report.json", report)
    return report


def import_cvat(xml: Path, manifest: dict, review: dict, evidence: dict) -> dict:
    """Read explicit keyframes only, with hash-bound reviewer attestation of exhaustive frames.

    reviewed_frames entries: {frame_id: int, labels: [str], exhaustive: true}.
    All visible boxes in those frame/label pairs must have reviewed=true and keyframe=1.
    """
    validate_input(evidence)
    if manifest.get("schema") != "cvat-source-manifest/v1" or manifest["input_id"] != digest(
        evidence
    ):
        raise ValueError("Source manifest does not match immutable harness input")
    if review.get("schema") != "cvat-review/v1" or review.get("manifest_id") != digest(manifest):
        raise ValueError("Review is not bound to this source manifest")
    if review.get("annotations_sha256") != sha256(Path(xml)):
        raise ValueError("Review must attest the exact edited XML hash")
    if not all(
        isinstance(review.get(k), str) and review[k].strip() for k in ("reviewer", "reason")
    ):
        raise ValueError("Explicit reviewer and reason required")
    frames = {f["frame_id"]: f for f in evidence["tables"]["frames"]}
    source_to_frame = {f["source_frame"]: f for f in frames.values()}
    coverage = set()
    for entry in review["reviewed_frames"]:
        if entry.get("exhaustive") is not True or entry["frame_id"] not in frames:
            raise ValueError("Only explicitly exhaustive source frames may be scored")
        if not entry["labels"] or not set(entry["labels"]) <= set(manifest["labels"]):
            raise ValueError("Reviewed labels must be declared by the manifest")
        for role in entry["labels"]:
            pair = (entry["frame_id"], role)
            if pair in coverage:
                raise ValueError("Duplicate reviewed frame and label")
            coverage.add(pair)
    raw = Path(xml).read_bytes()
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("DTD and entity declarations are not supported")
    tree = ET.fromstring(raw)
    if tree.tag != "annotations" or tree.findtext("version") != "1.1":
        raise ValueError("Expected CVAT XML version 1.1")
    if tree.findall("image"):
        raise ValueError("Expected video tracks, not image annotations")
    detections = {d["detection_id"]: d for d in evidence["tables"]["detections"]}
    ground_truth, reviews, seen, seen_evidence = [], [], set(), set()
    for track in tree.findall("track"):
        role, tid = track.attrib["label"], track.attrib["id"]
        if any(shape.tag != "box" for shape in track):
            raise ValueError("Only box tracks are supported for exhaustive tracking review")
        boxes = track.findall("box")
        explicit = {int(b.attrib["frame"]): b for b in boxes}
        if len(explicit) != len(boxes):
            raise ValueError("Duplicate box frame within a track")
        # CVAT interpolation may imply a visible box despite no explicit box at this frame.
        for frame_id, label in coverage:
            if role != label:
                continue
            sf = frames[frame_id]["source_frame"]
            preceding = [f for f in explicit if f <= sf]
            if preceding and sf not in explicit and explicit[max(preceding)].get("outside") == "0":
                raise ValueError(
                    "Reviewed frame contains an interpolated box; add explicit keyframe"
                )
        for sf, box in explicit.items():
            frame = source_to_frame.get(sf)
            if (
                frame is None
                or (frame["frame_id"], role) not in coverage
                or box.get("outside") == "1"
            ):
                continue
            attrs = {a.attrib["name"]: a.text or "" for a in box.findall("attribute")}
            if attrs.get("reviewed") != "true" or box.get("keyframe") != "1":
                raise ValueError("Exhaustive frame contains unreviewed or interpolated shape")
            key = (frame["frame_id"], role, tid)
            if key in seen:
                raise ValueError("Duplicate ground-truth identity in a frame")
            seen.add(key)
            bbox = [float(box.attrib[k]) for k in ("xtl", "ytl", "xbr", "ybr")]
            if not all(math.isfinite(v) for v in bbox) or not (
                0 <= bbox[0] < bbox[2] <= frame["width"]
                and 0 <= bbox[1] < bbox[3] <= frame["height"]
            ):
                raise ValueError("Reviewed box is invalid or outside original image")
            did = attrs.get("detection_id", "")
            if did and (did not in detections or detections[did]["frame_id"] != frame["frame_id"]):
                raise ValueError("Shape evidence ID does not belong to its source frame")
            if did and did in seen_evidence:
                raise ValueError("Detection evidence referenced by multiple reviewed boxes")
            if did:
                seen_evidence.add(did)
            ground_truth.append(
                {
                    "frame_id": frame["frame_id"],
                    "source_frame": sf,
                    "timestamp_s": frame["timestamp_s"],
                    "track_id": tid,
                    "label": role,
                    "bbox": bbox,
                    "reviewed": True,
                    "detection_id": did or None,
                }
            )
            if not did:
                continue  # New detections have no original evidence ID for identity/team review.
            end = min(evidence["window"]["end_s"], math.nextafter(frame["timestamp_s"], math.inf))
            for field, value in (
                ("identity", attrs.get("identity", "")),
                ("team", attrs.get("team", "unknown")),
            ):
                if value in ("", "unknown"):
                    continue
                if field == "team":
                    if value not in ("0", "1"):
                        raise ValueError("Reviewed team must be anonymous cluster 0 or 1")
                    value = int(value)
                correction = {
                    "field": field,
                    "value": value,
                    "start_s": frame["timestamp_s"],
                    "end_s": end,
                    "reviewer": review["reviewer"],
                    "reason": f"{review['reason']} [CVAT XML sha256:{sha256(Path(xml))}]",
                    "evidence_ids": [did],
                    "tracklet_id": detections[did]["tracklet_id"],
                }
                validate_review(correction, evidence)
                reviews.append(correction)
    return {
        "schema": "reviewed-tracking/v1",
        "source_sha256": evidence["source_sha256"],
        "input_id": digest(evidence),
        "manifest_id": digest(manifest),
        "review": review,
        "ground_truth": ground_truth,
        "harness_reviews": reviews,
        "limitations": [
            "Image-space boxes are tracking ground truth; geometry is not recalibrated",
            "Identity and team reviews apply only to explicitly reviewed sample times",
            "Ball candidates are not included in this player-tracking exchange",
        ],
    }


def apply_harness_reviews(harness, run_id: str, imported: dict) -> dict:
    """Apply supported corrections as append-only revisions, preserving earlier results."""
    context = harness.context(run_id)
    if context["input_id"] != imported["input_id"]:
        raise ValueError("Review belongs to another harness input")
    evidence = harness.get(context["input_id"])
    for review in imported["harness_reviews"]:
        validate_review(review, evidence)
    if not imported["harness_reviews"]:
        return {"count": 0, "revision": context["revision"], "reused": False}
    return harness.review_batch(run_id, imported["harness_reviews"], source_key=digest(imported))
