# Annotation and tracking evaluation

The CVAT adapter exports existing video detections as **unreviewed suggestions**. The demo export at `artifacts/integrations/cvat-demo-v2/` contains 2,318 suggestions across 100 sampled frames and zero reviewed boxes. Its accuracy fields remain null until human review. No annotations have been uploaded to a CVAT server.

## Review round trip

Use `soccerviz integrations --store artifacts/harness cvat-export RUN_ID --out artifacts/cvat/new-export` with an existing harness run. The export contains CVAT XML 1.1, a source-clock manifest, an empty review template, and a machine-readable readiness report.

Create a CVAT video task from the **original source video**, starting at frame zero with frame step one. Import `annotations.xml` using “CVAT for video 1.1.” Labels and attributes are declared in the XML; when creating the task, configure matching labels and these mutable attributes: `reviewed` checkbox (false), `detection_id` text (empty), `identity` text (empty), and `team` select (`unknown`, `0`, `1`). Bounding boxes use original image pixels; the generated 960×540 preview is unsuitable for this task.

XML frame numbers are original decoded frame ordinals, not sampled frame IDs. `manifest.json` preserves both, plus source PTS/time base, timestamps, dimensions, immutable harness input hash, video hash, and model hashes. Outside markers end each suggestion before unsampled gaps and after its last observation. This prevents CVAT from interpolating predictions across missing evidence. Human-created tracks must use explicit keyframes at every reviewed sample; this adapter never synthesizes ground truth by interpolation.

Review every visible object of the selected class in each selected sample: fix boxes/track identities, add misses, delete false positives, and set every remaining box's `reviewed` attribute to true. Preserve `detection_id` only for a shape corresponding to that original observation; new boxes leave it empty. Optional `identity` and anonymous `team` corrections become harness reviews. Track IDs in CVAT represent evaluation identities and need not equal model tracklet IDs.

Export the edited CVAT XML. Fill a copy of `review-template.json`:

```json
{
  "schema": "cvat-review/v1",
  "manifest_id": "copy the generated template value",
  "annotations_sha256": "SHA256 of the exact reviewed XML file",
  "reviewer": "actual reviewer name",
  "reason": "what was reviewed and corrected",
  "reviewed_frames": [
    {"frame_id": 0, "labels": ["player"], "exhaustive": true}
  ]
}
```

`frame_id` here is the sampled SoccerViz ID from the manifest. An exhaustive frame can legitimately contain zero objects: deleting all suggestions and attesting that frame provides an explicit negative. Unreviewed boxes, implicit/interpolated shapes, malformed coordinates, duplicate identities, and mismatched source evidence are rejected. The manifest/sidecar hashes establish data lineage; this offline workflow trusts the named human's attestation and does not authenticate a CVAT account.

Import with `soccerviz integrations --store artifacts/harness cvat-import RUN_ID --xml reviewed.xml --manifest manifest.json --review review.json --out reviewed-tracking.json`. Add `--preview` to validate/export without applying corrections. Supported identity/team corrections form an atomic, idempotent review batch; the full reviewed tracking artifact is attached to the run. Review intervals target only explicitly reviewed sample timestamps, preserving unknowns between them. Image-space box changes do not silently recalibrate pitch coordinates. Newly added boxes and corrected box geometry are available for tracking evaluation; downstream perception re-projection is a separate integration step. Ball candidates are not yet included.

## Official TrackEval

Create the persistent evaluation environment from the tested requirements, which pin the official upstream source commit:

```sh
python scripts/setup/setup_integrations.py evaluation
```

This installs `.venvs/evaluation/bin/python` using `envs/evaluation/requirements.txt`, including official TrackEval commit `12c8791b303e0a0b50f753af204249e622d0281a`. The core environment remains separate. To verify the adapter with this runtime:

```sh
PYTHONPATH=src .venvs/evaluation/bin/python -m pytest tests/test_annotation_adapter.py tests/test_tracking_evaluation.py tests/test_integrations_cli.py -q
```

Upstream declares NumPy/SciPy dependencies and installs with the project's current core stack. Its HOTA and Identity implementations still use removed `np.float`/`np.int` aliases. A lock-protected compatibility proxy supplies exactly those builtin aliases within the two metric modules during evaluation, then restores their globals. Shared NumPy is unchanged and the compatibility treatment is recorded in the report. The PyPI `trackeval==1.3.0` distribution identifies a different repository/fork and is not the tested dependency.

Run `soccerviz integrations --store artifacts/harness evaluate RUN_ID --reviewed reviewed-tracking.json --out tracking-report.json --python .venvs/evaluation/bin/python`. Default predictions come from the run's frozen detections. An alternative `--predictions` JSON must contain `source_sha256` and `predictions`, whose rows have `frame_id`, `track_id`, `label`, and `bbox: [x0,y0,x1,y1]`. Compare trackers on the same detections and identical reviewed coverage. The source hash must match the video. Reports hash both predictions and reviewed inputs.

The adapter calls upstream `HOTA.eval_sequence` and `Identity.eval_sequence` with original-pixel bounding-box IoU. Scores are 0–1. HOTA averages IoU thresholds 0.05–0.95, and IDF1 uses 0.5 IoU. Classes are scored separately, only on exhaustively reviewed frames; false positives on explicitly empty frames count. Sparse sampled results are not full-video metrics, and no MOTChallenge-specific distractor preprocessing is applied.

Verification uses clearly marked synthetic fixtures, independently checking perfect tracking, an identity switch with unchanged detections, missed detections, explicit empty frames, source mismatches, and unreviewed/interpolated annotation rejection. Synthetic tests establish adapter correctness and do not establish real-video accuracy.

References: [CVAT's XML format](https://docs.cvat.ai/docs/manual/advanced/formats/format-cvat/) and [official TrackEval repository](https://github.com/JonathonLuiten/TrackEval).
