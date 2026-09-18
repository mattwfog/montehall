"""Execution invariants, including real subprocess controller restart boundaries."""

import io
import json
import os
import subprocess
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path

import pytest

from soccerviz.providers.execution import (
    _REMOTE,
    _RUNNER,
    RESULT,
    JobSpec,
    RemoteError,
    SparkExecutor,
    TransportError,
    build_spec,
    canonical,
    immutable_json,
    prefect_flow,
    publish_archive,
    validate_result,
)

IMAGE = "sha256:" + "a" * 64


def spec_for(tmp_path, **kwargs):
    source = tmp_path / "input.txt"
    source.write_text("observations")
    files = {"data/input.txt": source}
    return build_spec(IMAGE, ["python", "model.py"], files, **kwargs), files


def archive_for(spec, data=b"{}", *, checksum=None, extra=None):
    import hashlib

    result = {
        "job_id": spec.job_id,
        "status": "completed",
        "files": {"report.json": checksum or hashlib.sha256(data).hexdigest()},
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        entries = [("report.json", data), (RESULT, canonical(result))]
        if extra:
            entries.append(extra)
        for name, value in entries:
            member = tarfile.TarInfo(name)
            member.size = len(value)
            tar.addfile(member, io.BytesIO(value))
    buffer.seek(0)
    return buffer


def test_job_id_tracks_content_not_local_path(tmp_path):
    spec, files = spec_for(tmp_path)
    other = tmp_path / "relocated"
    other.write_bytes(next(iter(files.values())).read_bytes())
    same = build_spec(IMAGE, spec.command, {"data/input.txt": other})
    assert spec.job_id == same.job_id
    other.write_text("corrected")
    assert build_spec(IMAGE, spec.command, {"data/input.txt": other}).job_id != spec.job_id
    assert JobSpec.from_dict(json.loads(canonical(asdict(spec)))).job_id == spec.job_id


@pytest.mark.parametrize("path", ["../bad", "/bad", "a/../bad", "a//b", "."])
def test_input_path_rejected(path):
    with pytest.raises(ValueError):
        JobSpec(IMAGE, ("true",), {path: "a" * 64})


def test_image_tag_rejected():
    with pytest.raises(ValueError, match="Pin image"):
        JobSpec("soccerviz:latest", ("true",), {})


def test_immutable_manifest(tmp_path):
    path = tmp_path / "spec.json"
    immutable_json(path, {"a": 1})
    immutable_json(path, {"a": 1})
    with pytest.raises(RemoteError, match="Immutable"):
        immutable_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


def test_retrieval_integrity_atomicity_and_idempotency(tmp_path):
    spec, _ = spec_for(tmp_path)
    dest = tmp_path / "published"
    with pytest.raises(RemoteError, match="hash mismatch"):
        publish_archive(archive_for(spec, checksum="b" * 64), dest, spec)
    assert not dest.exists()
    publish_archive(archive_for(spec), dest, spec)
    assert validate_result(dest, spec)["job_id"] == spec.job_id
    publish_archive(io.BytesIO(b"not consumed because already verified"), dest, spec)
    (dest / "report.json").write_text("corrupted")
    with pytest.raises(RemoteError, match="hash mismatch"):
        publish_archive(archive_for(spec), dest, spec)


@pytest.mark.parametrize(
    "extra", [("unexpected.txt", b"x"), ("../escaped", b"x"), ("report.json", b"x")]
)
def test_bad_archive_never_published(tmp_path, extra):
    spec, _ = spec_for(tmp_path)
    dest = tmp_path / "published"
    with pytest.raises((ValueError, RemoteError)):
        publish_archive(archive_for(spec, extra=extra), dest, spec)
    assert not dest.exists()
    assert not (tmp_path / "escaped").exists()


# Fake only Docker, leaving the complete remote-control program and subprocess
# boundaries real. Each call reads persistent state, as a fresh caller would.
_FAKE_DOCKER = r"""#!/usr/bin/env python3
import json, os, pathlib, sys
p = pathlib.Path(os.environ['FAKE_DOCKER_STATE'])
db = json.loads(p.read_text()) if p.exists() else {}
a = sys.argv[1:]
cmd = a[0]
if cmd == 'info': print('daemon')
elif cmd == 'inspect':
    if a[1] not in db:
        print('Error: No such object: ' + a[1], file=sys.stderr); sys.exit(1)
    print(json.dumps([db[a[1]]]))
elif cmd == 'create':
    name = a[a.index('--name') + 1]
    labels = dict(a[i+1].split('=', 1) for i,v in enumerate(a) if v == '--label')
    if name in db: sys.exit(1)
    db[name] = {'Id': name, 'State': {'Status': 'created', 'ExitCode': 0},
                'Config': {'Labels': labels}, 'starts': 0}
    p.write_text(json.dumps(db)); print(name)
elif cmd == 'start':
    marker = pathlib.Path(str(p) + '.fail-start')
    if marker.exists(): marker.unlink(); sys.exit(1)
    db[a[1]]['State']['Status'] = 'running'
    db[a[1]]['starts'] += 1
    p.write_text(json.dumps(db)); print(a[1])
elif cmd == 'ps':
    label = a[a.index('--filter') + 1].split('=',1)[1]
    k,v = label.split('=',1)
    print('\n'.join(n for n,o in db.items() if o['State']['Status']=='running'
        and o['Config']['Labels'].get(k)==v))
elif cmd == 'logs': print('model failed')
else: sys.exit(2)
"""


@pytest.fixture
def controller(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    executable = fake_bin / "docker"
    executable.write_text(_FAKE_DOCKER)
    executable.chmod(0o755)
    state = tmp_path / "docker.json"
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(state))
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ["PATH"])
    remote = tmp_path / "remote"

    def call(mode, spec, files=None):
        import base64

        request = {
            "mode": mode,
            "root": str(remote),
            "spec": asdict(spec),
            "job_id": spec.job_id,
            "max_concurrency": 1,
            "runner": _RUNNER,
        }
        buffer = io.BytesIO()
        if files is not None:
            with tarfile.open(fileobj=buffer, mode="w") as tar:
                for name, path in files.items():
                    tar.add(path, arcname=name)
        process = subprocess.run(
            [sys.executable, "-c", _REMOTE, base64.b64encode(canonical(request)).decode()],
            input=buffer.getvalue(),
            capture_output=True,
            check=False,
        )
        if process.returncode:
            raise RemoteError(process.stderr.decode())
        return json.loads(process.stdout)

    return call, state, remote


def test_crash_between_create_and_start_resumes_once(tmp_path, controller):
    call, state, _ = controller
    spec, files = spec_for(tmp_path)
    call("stage", spec, files)
    Path(str(state) + ".fail-start").touch()
    with pytest.raises(RemoteError):
        call("submit", spec)
    assert call("check", spec)["status"] == "created"
    assert call("submit", spec)["status"] == "running"
    # A fresh controller subprocess after a lost response must not start twice.
    assert call("submit", spec)["status"] == "running"
    db = json.loads(state.read_text())
    assert db["soccerviz-" + spec.job_id]["starts"] == 1


def test_remote_concurrency_and_terminal_failure_no_rerun(tmp_path, controller):
    call, state, _ = controller
    spec, files = spec_for(tmp_path)
    second = JobSpec.from_dict({**asdict(spec), "attempt": 1})
    call("stage", spec, files)
    call("stage", second, files)
    call("submit", spec)
    assert call("submit", second)["status"] == "busy"
    db = json.loads(state.read_text())
    db["soccerviz-" + spec.job_id]["State"] = {"Status": "exited", "ExitCode": 137}
    state.write_text(json.dumps(db))
    assert call("submit", spec)["status"] == "failed"
    assert json.loads(state.read_text())["soccerviz-" + spec.job_id]["starts"] == 1
    assert call("submit", second)["status"] == "running"


def test_remote_staging_rejects_changed_input(tmp_path, controller):
    call, _, remote = controller
    spec, files = spec_for(tmp_path)
    next(iter(files.values())).write_text("changed after hashing")
    with pytest.raises(RemoteError, match="hash mismatch"):
        call("stage", spec, files)
    assert not (remote / spec.job_id).exists()


def test_lost_ssh_response_has_recoverable_error(tmp_path, monkeypatch):
    spec, _ = spec_for(tmp_path)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 255, b"", b"lost")
    )
    with pytest.raises(TransportError):
        SparkExecutor(tmp_path).submit(spec)


def test_real_prefect_flow_retries_transient_and_reuses_job(tmp_path, monkeypatch):
    pytest.importorskip("prefect")
    from prefect.testing.utilities import prefect_test_harness

    import soccerviz.providers.execution as adapter

    spec, files = spec_for(tmp_path)
    counters = {"starts": 0, "checks": 0, "submit_calls": 0}

    class FakeExecutor:
        def __init__(self, **kwargs):
            pass

        def check(self, _):
            counters["checks"] += 1
            state = "missing" if counters["starts"] == 0 else "running"
            if counters["checks"] >= 4:
                state = "completed"
            return {"status": state}

        def stage(self, *args):
            pass

        def submit(self, _):
            counters["submit_calls"] += 1
            if counters["starts"] == 0:
                counters["starts"] = 1
                raise TransportError("caller lost response after container started")
            return {"status": "running"}

        def retrieve(self, _):
            return tmp_path / "verified-artifacts"

    monkeypatch.setattr(adapter, "SparkExecutor", FakeExecutor)
    with prefect_test_harness():
        run = prefect_flow(max_polls=3, poll_seconds=0)
        result = run(asdict(spec), {k: str(v) for k, v in files.items()}, {})
        assert result.endswith("verified-artifacts")
        assert counters["starts"] == 1
        assert counters["submit_calls"] == 2
        run(asdict(spec), {}, {})
        assert counters["starts"] == 1


def test_prefect_served_entrypoint_is_reloadable():
    pytest.importorskip("prefect")
    from prefect.deployments.runner import RunnerDeployment
    from prefect.flows import load_flow_from_entrypoint

    loaded = load_flow_from_entrypoint("scripts/spark/prefect_worker.py:run_spark_job")
    deployment = RunnerDeployment.from_flow(loaded, name="smoke")
    reloaded = load_flow_from_entrypoint(deployment.entrypoint)
    assert reloaded.name == "soccerviz-spark-deployment"
