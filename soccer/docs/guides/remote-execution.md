# Recoverable Spark execution with Prefect

SoccerViz owns immutable evidence and match revisions. Prefect owns orchestration,
bounded retries, polling, and an optional deployment process. The GPU workload is
a detached Docker container on Spark, so a laptop disconnect or Prefect worker
exit does not stop it. This adapter uses the existing GPU image; it installs no
service or package on Spark and does not alter existing Spark scripts.

Create the isolated execution runtime with the repository setup script:

```sh
.venv/bin/python scripts/setup/setup_integrations.py execution
```

This installs `envs/execution/requirements.txt` into `.venvs/execution`, including
Prefect 3.8.5 and pytest 9.1.1, without changing the core environment. The base
package can still stage, submit, check, and retrieve without Prefect. Run the
execution adapter with `PYTHONPATH=src .venvs/execution/bin/python` as below.

## Run a short video job

First inspect and pin the existing image ID:

```sh
ssh -o ClearAllForwardings=yes spark 'docker image inspect soccerviz-cv:0.2 --format "{{.Id}}"'
```

Then run from the repository root, substituting that ID:

```sh
PYTHONPATH=src .venvs/execution/bin/python scripts/spark/prefect_worker.py video \
  --image sha256:950bb67171f6985147f3f83d3407676587efeb754e187637e8ef148b80f63363 \
  --seconds 2 --hz 2
```

The script prints the stable job ID before execution. It explicitly uploads the
Python source files, source video, and three detector checkpoints, all hashed.
Inputs are mounted read-only. Outputs appear only after complete verification at
`artifacts/execution/JOB_ID/artifacts/`. The directory contains the existing video
Parquet/report format plus `__soccerviz_result.json` with every output hash.

Use the returned directory with the existing harness:

```sh
.venv/bin/soccerviz harness start --video-run artifacts/execution/JOB_ID/artifacts \
  --start-s 0 --end-s 1
```

Resume from a fresh process after interruption; the original local input files
are unnecessary once remote staging completed:

```sh
PYTHONPATH=src .venvs/execution/bin/python scripts/spark/prefect_worker.py check JOB_ID
PYTHONPATH=src .venvs/execution/bin/python scripts/spark/prefect_worker.py resume JOB_ID
PYTHONPATH=src .venvs/execution/bin/python scripts/spark/prefect_worker.py retrieve JOB_ID
```

Retain the same `--store`, `--host`, and `--remote-root` options when resuming. A
local immutable `remote.json` records the binding. If interruption occurred before
staging completed, rerun the original `video` command with the same input bytes.
A failed workload remains failed; `--attempt 1` explicitly creates another job.
No failed GPU computation is silently rerun by transport retry logic.

## Specialist API

```python
from dataclasses import asdict
from soccerviz.providers.execution import build_spec, prefect_flow

files = {"src/model.py": "/absolute/path/model.py", "data/input.json": "/absolute/path/input.json"}
spec = build_spec(
    image="sha256:" + pinned_image_hash,
    command=["python", "src/model.py", "--out", "/output/report.json"],
    files=files,
    required_outputs=["report.json"],
)
artifact_directory = prefect_flow()(
    asdict(spec), files,
    {"store": "/absolute/path/artifacts/execution", "host": "spark",
     "remote_root": "~/soccerviz-execution", "max_concurrency": 1},
)
```

The job identity includes image digest, command arguments, input relative paths
and SHA256s, required outputs, CPU/GPU settings, explicit attempt, and protocol
version. Local absolute input locations do not affect identity. Any change to the
container-runner contract requires a protocol-version bump. Do not put secrets in
commands or uploaded input manifests; they are intentionally inspectable.

`SparkExecutor.stage`, `.submit`, `.check`, and `.retrieve` expose the same
operations without Prefect. No method contains a retry loop. The flow retries
transport errors and active/busy states within fixed bounds. Terminal execution,
missing-output, and integrity failures fail immediately. Default completion
polling is 120 retries at five seconds; expiry leaves the remote job intact for
later inspection/resumption. Configure `prefect_flow(max_polls=..., poll_seconds=...)`
for longer jobs. Poll delay is bounded to 60 seconds and retries to 10,000.

## Durable deployment

An ad-hoc flow uses Prefect's temporary local server. The remote computation and
SoccerViz manifests survive its exit, but durable orchestration history and
scheduled continuation require a persistent Prefect API/database and a supervised
worker or served deployment. Start those explicitly in the deployment environment:

```sh
.venvs/execution/bin/prefect server start
# In the worker's environment:
export PREFECT_API_URL=http://127.0.0.1:4200/api
PYTHONPATH=src .venvs/execution/bin/python scripts/spark/prefect_worker.py serve --name soccerviz-spark
```

The served flow accepts `spec_data`, `files`, and `options` parameters as above.
The worker must access the local input files before first staging and the SSH
identity for Spark. The served flow limits concurrent flow runs to one by default.
An additional flock-protected remote guard limits concurrently running containers
in this adapter's remote-root namespace. Use one shared remote root and consistent
`max_concurrency` across workers. It does not count unrelated GPU workloads; choose
capacity appropriate to the host. Process work pools are an alternative for
centrally managed infrastructure and concurrency.

Prefect documents [workers](https://docs.prefect.io/v3/concepts/workers),
[work pools](https://docs.prefect.io/v3/concepts/work-pools), and
[bounded task retries](https://docs.prefect.io/v3/concepts/tasks). This adapter uses
those orchestration concepts while keeping evidence registration in SoccerViz.

## Recovery and integrity guarantees

- Inputs stage into a temporary directory; exact membership and hashes must pass
  before the immutable remote job directory is renamed into place.
- `docker create` uses the stable job name. A crash before `docker start` leaves a
  recoverable `created` container. A lost start response is resolved by inspection.
- Containers are not automatically removed. Their terminal state prevents retries
  from rerunning computations. Do not prune managed containers while jobs or
  evidence are still active; lifecycle cleanup is deliberately explicit.
- Container completion requires exit code zero, every required output, and an
  output manifest. A killed container with no manifest is failed, not successful.
- Retrieval rejects links, traversal, duplicates, unexpected files, missing files,
  and hash mismatches. The final artifact directory appears atomically after all
  checks. Existing retrieved files are revalidated on every reuse.
- SSH calls have a five-minute transfer/control timeout. A timeout leaves job
  state uncertain and raises a retryable transport error rather than declaring
  remote failure. Large uploads may need a future chunked content store.

Tests execute the remote controller in independent subprocesses against a fake
Docker daemon to exercise actual filesystem/process recovery boundaries, and an
optional real Prefect test verifies retries after a lost submit response. These
are distinct from GPU accuracy evaluation; successful execution says nothing
about detector or tracking quality.

## Verified integration run (2026-09-07)

Prefect 3.8.5 ran the existing pinned Spark CV image on one second of the public
video at 2 Hz. Job `31f8bdc943c789c34d456f12866bb5f8dd438e8d8e2b887a56d5aed4a6d78210`
completed with two frames and 48 detections; ten output files passed SHA256
verification. A second, fresh Prefect process resumed the same job and retained
identical execution start/end timestamps and outputs. The machine-readable
record is `results/experiments/spark-execution-report.json` (also retained at
`artifacts/execution/smoke-verification.json`).

The served deployment entrypoint was loaded and reloaded through Prefect's real
deployment API. A persistent server/served worker was not installed or left
running. Full workload capacity, worker supervision, and long-duration GPU
recovery remain deployment checks rather than claims established by this smoke.


Run all execution checks in the installed integration runtime:

```sh
PYTHONPATH=src PREFECT_HOME=/private/tmp/soccerviz-prefect-tests \
  PREFECT_SERVER_ANALYTICS_ENABLED=false \
  .venvs/execution/bin/python -m pytest tests/test_execution_adapter.py -q
```

The optional real Prefect test starts a temporary local server; its local listening
port must be permitted by the execution environment. No persistent server remains.
