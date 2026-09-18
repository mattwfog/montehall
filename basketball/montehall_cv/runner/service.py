"""Spark-resident CV runner (design D11).

Queue = the S3 manifest itself: a job is pending when
input/<job_id>/basketball/video.mp4 exists in the raw bucket and no
outputs/<job_id>/ object exists in the inferences bucket. No queue infra,
idempotent by construction, resumable at every layer (S3 existence checks,
per-stage _SUCCESS markers, incremental parquet).

Status events (processing/progress/done/error, the job_watcher contract)
are emitted to Redis streams when MONTEHALL_REDIS_URL is set and skipped
cleanly when not — S3 state remains the source of truth either way, so the
app can reconcile jobs that completed while Redis was unreachable.

Env: CV_S3_ENDPOINT, CV_S3_ACCESS_KEY, CV_S3_SECRET_KEY, CV_RAW_BUCKET,
CV_INFERENCES_BUCKET, CV_OUT_ROOT, CV_COURT_WEIGHTS, CV_DETECTOR_WEIGHTS
(optional), CV_REID_WEIGHTS / CV_LEGIBILITY_WEIGHTS / CV_OCR_WEIGHTS
(optional), CV_RENDER=1 + CV_RENDER_SCALE (optional annotated video,
uploaded and emitted as the done event's video_key -> the app's
output_video_s3_key), CV_GPU_GATE=1 (skip poll cycles while another
process owns the GPU — training saturates the CUDA allocator and any
extract would OOM straight into the poison-park path),
MONTEHALL_REDIS_URL (optional), ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
import traceback
from pathlib import Path

STATUS_STREAM = "cv:events:status"
POLL_SECONDS = 60
MAX_FAILURES = 3  # consecutive failures before a job is parked with an S3 error marker


class Runner:
    def __init__(self) -> None:
        import boto3

        self._s3 = boto3.client(
            "s3",
            endpoint_url=_env("CV_S3_ENDPOINT"),
            aws_access_key_id=_env("CV_S3_ACCESS_KEY"),
            aws_secret_access_key=_env("CV_S3_SECRET_KEY"),
        )
        self._raw_bucket = _env("CV_RAW_BUCKET")
        self._inf_bucket = _env("CV_INFERENCES_BUCKET")
        self._out_root = Path(_env("CV_OUT_ROOT"))
        self._court_weights = Path(_env("CV_COURT_WEIGHTS"))
        detector = os.environ.get("CV_DETECTOR_WEIGHTS")
        self._detector_weights = Path(detector) if detector else None
        reid = os.environ.get("CV_REID_WEIGHTS")
        self._reid_weights = Path(reid) if reid else None
        legibility = os.environ.get("CV_LEGIBILITY_WEIGHTS")
        self._legibility_weights = Path(legibility) if legibility else None
        ocr = os.environ.get("CV_OCR_WEIGHTS")
        self._ocr_weights = Path(ocr) if ocr else None
        self._long_gap = os.environ.get("CV_LONG_GAP", "") == "1"
        self._render = os.environ.get("CV_RENDER", "") == "1"
        self._render_scale = float(os.environ.get("CV_RENDER_SCALE", "1.0"))
        self._gpu_gate = os.environ.get("CV_GPU_GATE", "") == "1"
        self._redis = self._connect_redis()
        self._failures: dict[str, int] = {}

    def _connect_redis(self):
        url = os.environ.get("MONTEHALL_REDIS_URL")
        if not url:
            return None
        import redis

        client = redis.Redis.from_url(url, decode_responses=True)
        client.ping()
        return client

    def pending_jobs(self) -> list[str]:
        raw_jobs = set()
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._raw_bucket, Prefix="input/"):
            for obj in page.get("Contents", []):
                parts = obj["Key"].split("/")
                if len(parts) >= 4 and parts[3] == "video.mp4":
                    raw_jobs.add(parts[1])
        done_jobs = set()
        for page in paginator.paginate(Bucket=self._inf_bucket, Prefix="outputs/"):
            for obj in page.get("Contents", []):
                parts = obj["Key"].split("/")
                if len(parts) >= 2:
                    done_jobs.add(parts[1])
        return sorted(raw_jobs - done_jobs)

    def process(self, job_id: str) -> dict:
        from montehall_cv.pipeline.run_all import run_all
        from montehall_cv.pipeline.video import probe
        from montehall_cv.runner.adapter import build_results

        self._emit({"job_id": job_id, "status": "processing"})
        video_path = self._out_root / job_id / "video.mp4"
        try:
            if not video_path.exists():
                video_path.parent.mkdir(parents=True, exist_ok=True)
                self._s3.download_file(
                    self._raw_bucket,
                    f"input/{job_id}/basketball/video.mp4",
                    str(video_path),
                )
            roster_path = self._fetch_roster(job_id)
            stages = run_all(
                video_path,
                self._out_root,
                job_id,
                self._court_weights,
                detector_weights=self._detector_weights,
                reid_weights=self._reid_weights,
                legibility_weights=self._legibility_weights,
                ocr_weights=self._ocr_weights,
                roster_map=roster_path,
                long_gap=self._long_gap,
                render=self._render,
                render_scale=self._render_scale,
            )
            info = probe(video_path)
            results = build_results(self._out_root / job_id, info.average_fps)
            stamp = datetime.datetime.now(datetime.UTC).strftime("%y%m%d_%H%M")
            evidence_keys = self._upload_evidence(job_id, stamp)
            if evidence_keys:
                # entity_id -> S3 key; the app presigns these for the
                # "Who is this?" modal (transformer ignores unknown keys)
                results["entity_evidence"] = evidence_keys
            tracks_key = self._upload_tracks(job_id, stamp)
            if tracks_key:
                # overlay keyframe tracks (B1) — app presigns for the
                # in-video naming prompts
                results["entity_tracks"] = tracks_key
            results_key = f"outputs/{job_id}/basketball/inference_outputs_{stamp}.json"
            self._s3.put_object(
                Bucket=self._inf_bucket,
                Key=results_key,
                Body=json.dumps(results).encode(),
                ContentType="application/json",
            )
            video_key = self._upload_annotated(job_id, stamp)
            self._emit(
                {
                    "job_id": job_id,
                    "status": "done",
                    "results_key": results_key,
                    "video_key": video_key,
                }
            )
            return {
                "job_id": job_id, "results_key": results_key,
                "video_key": video_key, "stages": stages,
            }
        except Exception as exc:
            self._emit({"job_id": job_id, "status": "error", "error": str(exc)[:500]})
            raise

    def _upload_annotated(self, job_id: str, stamp: str) -> str:
        """Ship the rendered overlay video when the stage produced one.
        Its key rides the done event as video_key -> the app stores it as
        output_video_s3_key and presigns it from the inferences bucket."""
        annotated = self._out_root / job_id / "annotated" / "annotated.mp4"
        if not annotated.exists():
            return ""
        video_key = f"outputs/{job_id}/basketball/annotated_{stamp}.mp4"
        self._s3.upload_file(
            str(annotated), self._inf_bucket, video_key,
            ExtraArgs={"ContentType": "video/mp4"},
        )
        return video_key

    def _upload_evidence(self, job_id: str, stamp: str) -> dict[str, str]:
        """Ship entity evidence sheets (A6): entity_id -> uploaded key."""
        stage_dir = self._out_root / job_id / "entity_evidence"
        keys: dict[str, str] = {}
        for jpg in sorted(stage_dir.glob("e*.jpg")):
            key = f"outputs/{job_id}/basketball/entity_evidence_{stamp}/{jpg.name}"
            self._s3.upload_file(
                str(jpg), self._inf_bucket, key,
                ExtraArgs={"ContentType": "image/jpeg"},
            )
            keys[jpg.stem[1:]] = key  # "e50.jpg" -> "50"
        return keys

    def _upload_tracks(self, job_id: str, stamp: str) -> str:
        """Ship the overlay keyframe tracks (B1) when the stage produced them."""
        tracks = self._out_root / job_id / "entity_tracks" / "tracks.json"
        if not tracks.exists():
            return ""
        key = f"outputs/{job_id}/basketball/entity_tracks_{stamp}.json"
        self._s3.upload_file(
            str(tracks), self._inf_bucket, key,
            ExtraArgs={"ContentType": "application/json"},
        )
        return key

    def _gpu_busy(self) -> bool:
        """Another process owns the GPU (training saturates the CUDA
        allocator — 2026-07-08 lesson: even a ResNet34 .to(cuda) OOMs).
        nvidia-smi absent or failing -> not busy (don't wedge the queue
        on a tooling problem)."""
        if not self._gpu_gate:
            return False
        import subprocess

        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:
            return False
        if out.returncode != 0:
            return False
        lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
        return bool(lines)

    def _fetch_roster(self, job_id: str) -> Path | None:
        """input/<job_id>/roster.json, written by the api at upload completion.
        Absent (older uploads, rosterless teams) -> attribution runs unfiltered."""
        roster_path = self._out_root / job_id / "roster.json"
        if roster_path.exists():
            return roster_path
        try:
            self._s3.download_file(
                self._raw_bucket, f"input/{job_id}/roster.json", str(roster_path)
            )
            return roster_path
        except Exception:
            return None

    def _emit(self, event: dict) -> None:
        if self._redis is None:
            return
        try:
            self._redis.xadd(STATUS_STREAM, event)
        except Exception:
            print(f"redis emit failed (continuing): {traceback.format_exc(limit=1)}", flush=True)

    def _park(self, job_id: str, error: str) -> None:
        """Write an error marker so the S3-manifest queue stops re-polling this
        job (pending = input present AND no outputs/<job>/ object exists). A
        human deletes outputs/<job>/error.json to re-queue it."""
        self._s3.put_object(
            Bucket=self._inf_bucket,
            Key=f"outputs/{job_id}/error.json",
            Body=json.dumps(
                {"job_id": job_id, "status": "failed", "failures": MAX_FAILURES, "error": error[:1000]}
            ).encode(),
            ContentType="application/json",
        )
        self._emit({"job_id": job_id, "status": "failed", "parked": "true", "error": error[:500]})

    def loop(self, once: bool = False, job_id: str | None = None) -> None:
        while True:
            if self._gpu_busy():
                print("gpu busy (training run holds the allocator); waiting", flush=True)
                if once:
                    return
                time.sleep(POLL_SECONDS)
                continue
            jobs = [job_id] if job_id else self.pending_jobs()
            print(f"pending jobs: {len(jobs)}", flush=True)
            for jid in jobs:
                print(f"processing {jid}", flush=True)
                try:
                    summary = self.process(jid)
                    print(json.dumps(summary), flush=True)
                    self._failures.pop(jid, None)
                except Exception:
                    n = self._failures.get(jid, 0) + 1
                    self._failures[jid] = n
                    print(f"job {jid} failed ({n}/{MAX_FAILURES}):\n{traceback.format_exc()}", flush=True)
                    if n >= MAX_FAILURES:
                        # Stop re-buying calls on a job that keeps dying: park it
                        # with an error marker so the manifest no longer lists it.
                        try:
                            self._park(jid, traceback.format_exc(limit=3))
                            print(f"job {jid} parked after {n} failures", flush=True)
                        except Exception:
                            print(f"could not park {jid}:\n{traceback.format_exc()}", flush=True)
                        self._failures.pop(jid, None)
            if once:
                return
            # A forced --job-id that has succeeded or been parked will not recur
            # via the manifest, so stop instead of spinning on it forever.
            if job_id is not None and job_id not in self._failures:
                return
            time.sleep(POLL_SECONDS)


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} not set")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--job-id", default=None, help="force one job regardless of manifest")
    args = parser.parse_args()

    # Single-instance lock: two runners race per-job stages (2026-07-10: a
    # docker-exec'd runner survived its systemd unit's stop and raced a
    # manual one — the loser crashed on the winner's renamed render tmp).
    # flock releases on ANY process death; no stale-lock handling needed.
    import fcntl

    lock_path = Path(_env("CV_OUT_ROOT")) / ".runner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another runner instance holds the lock; exiting")
    Runner().loop(once=args.once, job_id=args.job_id)


if __name__ == "__main__":
    main()
