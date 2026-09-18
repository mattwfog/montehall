"""Recoverable SSH/Docker execution, optionally orchestrated by Prefect.

The remote container is the durable execution handle. A disconnected caller may
submit the identical manifest again without rerunning completed or failed work.
No Prefect installation is required on the GPU host.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

PROTOCOL = "soccerviz-execution-v1"
RESULT = "__soccerviz_result.json"


class RemoteError(RuntimeError):
    """A terminal execution or integrity error."""


class TransportError(RemoteError):
    """SSH response unknown; safe to recheck or resubmit the same job."""


class RemotePending(RemoteError):
    """Job still running; Prefect owns bounded polling."""


class RemoteBusy(RemotePending):
    """Remote managed-container concurrency limit reached."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if not value or value == "." or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"Expected canonical relative path: {value!r}")
    return path


@dataclass(frozen=True)
class JobSpec:
    """JSON contract: pinned image, command argv, input hashes, expected outputs."""

    image: str
    command: tuple[str, ...]
    inputs: dict[str, str]
    required_outputs: tuple[str, ...] = ("report.json",)
    gpu: bool = True
    cpus: int = 4
    attempt: int = 0
    protocol: str = PROTOCOL

    def __post_init__(self):
        if self.protocol != PROTOCOL:
            raise ValueError("Unsupported execution protocol")
        if not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", self.image):
            raise ValueError("Pin image to sha256 image ID or repository@sha256 digest")
        if not self.command or any(not isinstance(v, str) or "\0" in v for v in self.command):
            raise ValueError("command must be nonempty argv without NUL characters")
        if not 1 <= self.cpus <= 64 or self.attempt < 0:
            raise ValueError("Invalid CPU limit or attempt")
        for name, checksum in self.inputs.items():
            relative_path(name)
            if not re.fullmatch("[0-9a-f]{64}", checksum):
                raise ValueError("Invalid input SHA256")
        for name in self.required_outputs:
            relative_path(name)
            if name == RESULT:
                raise ValueError("Output name reserved for completion manifest")
        if not self.required_outputs:
            raise ValueError("At least one required output is needed")

    @property
    def job_id(self):
        return hashlib.sha256(canonical(asdict(self))).hexdigest()

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        data["command"] = tuple(data["command"])
        data["required_outputs"] = tuple(data["required_outputs"])
        return cls(**data)


def build_spec(image, command, files, required_outputs=("report.json",), **kwargs):
    """Hash explicit relative-name → local-path inputs (no implicit secret upload)."""
    return JobSpec(
        image=image,
        command=tuple(command),
        inputs={name: digest(path) for name, path in sorted(files.items())},
        required_outputs=tuple(required_outputs),
        **kwargs,
    )


def immutable_json(path, value):
    payload = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic hard-link publication: another caller can never observe partial JSON.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RemoteError(f"Immutable manifest mismatch: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def validate_result(directory, spec):
    """Verify exact artifact membership and hashes before returning any evidence."""
    result = json.loads((directory / RESULT).read_text())
    if result.get("job_id") != spec.job_id or result.get("status") != "completed":
        raise RemoteError("Completion manifest does not describe this successful job")
    files = result.get("files", {})
    if not set(spec.required_outputs).issubset(files):
        raise RemoteError("Required outputs missing from completion manifest")
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise RemoteError("Symbolic link in retrieved artifacts")
        if path.is_file() and path.relative_to(directory).as_posix() != RESULT:
            actual.add(path.relative_to(directory).as_posix())
    if actual != set(files):
        raise RemoteError("Artifact membership differs from completion manifest")
    for name, checksum in files.items():
        relative_path(name)
        if digest(directory / name) != checksum:
            raise RemoteError(f"Artifact hash mismatch: {name}")
    return result


def publish_archive(archive, destination, spec):
    """Extract files only, verify all hashes, then atomically expose the directory."""
    destination = Path(destination)
    if destination.exists():
        return validate_result(destination, spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".retrieving-") as scratch:
        staging = Path(scratch) / "artifacts"
        staging.mkdir()
        with tarfile.open(fileobj=archive, mode="r:*") as tar:
            seen = set()
            for member in tar:
                relative_path(member.name)
                if not member.isfile() or member.name in seen:
                    raise RemoteError("Archive contains a link, directory, or duplicate file")
                seen.add(member.name)
                target = staging / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
        result = validate_result(staging, spec)
        try:
            staging.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            validate_result(destination, spec)
        return result


# Executed inside the pinned container. Inputs are mounted read-only at /work.
_RUNNER = r"""
import hashlib, json, os, pathlib, subprocess, sys, time
root = pathlib.Path('/output')
spec = json.loads(pathlib.Path('/control/spec.json').read_text())
job_id = pathlib.Path('/control/job_id').read_text()
started = time.time()
code = subprocess.call(spec['command'], cwd='/work')
result = {'job_id': job_id, 'exit_code': code, 'started_at': started, 'finished_at': time.time()}
files = {}
try:
    if code:
        raise RuntimeError('Command failed with exit code %s' % code)
    for name in spec['required_outputs']:
        if not (root / name).is_file():
            raise RuntimeError('Required output missing: ' + name)
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise RuntimeError('Symlink output is unsupported')
        if p.is_file():
            with p.open('rb') as f:
                files[p.relative_to(root).as_posix()] = hashlib.file_digest(f, 'sha256').hexdigest()
    result.update(status='completed', files=files)
except Exception as e:
    result.update(status='failed', error=str(e), files={})
p = root / '__soccerviz_result.json.tmp'
with p.open('w') as f:
    json.dump(result, f, sort_keys=True)
    f.flush()
    os.fsync(f.fileno())
os.replace(p, root / '__soccerviz_result.json')
sys.exit(0 if result['status'] == 'completed' else (code or 1))
"""


# Pure standard-library control program, sent over SSH rather than installed on Spark.
_REMOTE = r"""
import base64, fcntl, hashlib, json, os, pathlib, shutil, subprocess, sys, tarfile, tempfile
request = json.loads(base64.b64decode(sys.argv[1]))
root = pathlib.Path(request['root']).expanduser().resolve()
root.mkdir(parents=True, exist_ok=True)
job_id = request['job_id']
spec = request['spec']
if hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != job_id:
    raise ValueError('Job ID mismatch')
job = root / job_id
name = 'soccerviz-' + job_id
owner = hashlib.sha256(str(root).encode()).hexdigest()[:20]
def docker(*args, check=True):
    p = subprocess.run(['docker', *args], capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError(p.stderr.strip())
    return p
def inspect():
    # A daemon/permission failure must not be mistaken for a missing container.
    docker('info', '--format', '{{.ID}}')
    p = docker('inspect', name, check=False)
    if p.returncode:
        if 'no such object' in p.stderr.lower() or 'no such container' in p.stderr.lower():
            return None
        raise RuntimeError(p.stderr.strip())
    obj = json.loads(p.stdout)[0]
    if obj['Config']['Labels'].get('soccerviz.job') != job_id:
        raise RuntimeError('Container name is owned by another job')
    return obj
def status():
    obj = inspect()
    if obj is None:
        return {'job_id': job_id, 'status': 'staged' if job.exists() else 'missing'}
    state = obj['State']
    result = {'job_id': job_id, 'container_id': obj['Id'], 'status': state['Status'],
              'exit_code': state['ExitCode']}
    if state['Status'] in ('exited', 'dead'):
        p = job / 'output' / '__soccerviz_result.json'
        if p.exists():
            result.update(json.loads(p.read_text()))
        else:
            result.update(status='failed', error='Container exited without completion manifest')
        if state['ExitCode'] != 0:
            result['status'] = 'failed'
        if result['status'] == 'failed':
            logs = docker('logs', '--tail', '40', name, check=False)
            result['logs'] = (logs.stdout + logs.stderr)[-12000:]
    return result
def validate_inputs(directory):
    for rel, checksum in spec['inputs'].items():
        p = directory / rel
        with p.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != checksum:
                raise ValueError('Input hash mismatch: ' + rel)
lock = (root / '.submit.lock').open('a')
with lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    mode = request['mode']
    if job.exists():
        saved = json.loads((job / 'control/spec.json').read_text())
        if saved != spec:
            raise ValueError('Immutable remote manifest mismatch')
    if mode == 'stage':
        if not job.exists():
            with tempfile.TemporaryDirectory(dir=root, prefix='.staging-') as scratch:
                stage = pathlib.Path(scratch) / 'job'
                (stage / 'inputs').mkdir(parents=True)
                (stage / 'output').mkdir()
                (stage / 'control').mkdir()
                seen = set()
                with tarfile.open(fileobj=sys.stdin.buffer, mode='r|') as tar:
                    for member in tar:
                        if member.name not in spec['inputs'] or not member.isfile() or member.name in seen:
                            raise ValueError('Unexpected archive member')
                        seen.add(member.name)
                        p = stage / 'inputs' / member.name
                        p.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as f, p.open('xb') as out:
                            shutil.copyfileobj(f, out)
                if seen != set(spec['inputs']):
                    raise ValueError('Input archive incomplete')
                validate_inputs(stage / 'inputs')
                (stage / 'control/spec.json').write_text(json.dumps(spec, sort_keys=True))
                (stage / 'control/job_id').write_text(job_id)
                (stage / 'control/runner.py').write_text(request['runner'])
                stage.rename(job)
        print(json.dumps({'job_id': job_id, 'status': 'staged'}))
    elif mode == 'submit':
        if not job.exists():
            raise ValueError('Job must be staged first')
        obj = inspect()
        if obj is None or obj['State']['Status'] == 'created':
            active = docker('ps', '-q', '--filter', 'label=soccerviz.owner=' + owner).stdout.split()
            if len(active) >= request['max_concurrency']:
                print(json.dumps({'job_id': job_id, 'status': 'busy'}))
                sys.exit(0)
            if obj is None:
                validate_inputs(job / 'inputs')
                args = ['create', '--name', name, '--label', 'soccerviz.job=' + job_id,
                        '--label', 'soccerviz.owner=' + owner, '--cpus', str(spec['cpus']),
                        '--shm-size', '1g', '--user', str(os.getuid()) + ':' + str(os.getgid()),
                        '--env', 'PYTHONPATH=/work/src', '--env', 'YOLO_CONFIG_DIR=/tmp/yolo',
                        '--env', 'HOME=/tmp', '--env', 'OMP_NUM_THREADS=' + str(spec['cpus']),
                        '--mount', 'type=bind,src=' + str(job / 'inputs') + ',dst=/work,readonly',
                        '--mount', 'type=bind,src=' + str(job / 'control') + ',dst=/control,readonly',
                        '--mount', 'type=bind,src=' + str(job / 'output') + ',dst=/output',
                        '--workdir', '/work', '--entrypoint', 'python']
                if spec['gpu']:
                    args += ['--gpus', 'all']
                docker(*args, spec['image'], '/control/runner.py')
            # Crash between create/start leaves a recoverable created container.
            docker('start', name)
        print(json.dumps(status()))
    elif mode == 'check':
        print(json.dumps(status()))
    elif mode == 'fetch':
        result = status()
        if result['status'] != 'completed':
            raise ValueError('Cannot fetch an incomplete/failed job')
        with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as tar:
            for p in sorted((job / 'output').rglob('*')):
                if p.is_symlink():
                    raise ValueError('Symlink output is unsupported')
                if p.is_file():
                    tar.add(p, arcname=p.relative_to(job / 'output').as_posix(), recursive=False)
    else:
        raise ValueError('Unknown operation')
"""


class SparkExecutor:
    """No internal retries: Prefect or the calling operator owns retry policy."""

    def __init__(self, store, host="spark", remote_root="~/soccerviz-execution", max_concurrency=1):
        if not re.fullmatch(r"[a-zA-Z0-9_.@-]+", host) or host.startswith("-"):
            raise ValueError("Invalid SSH host")
        if not 1 <= max_concurrency <= 16:
            raise ValueError("Concurrency must be between 1 and 16")
        if "," in remote_root or "\n" in remote_root:
            raise ValueError("Invalid remote bind-mount path")
        self.store, self.host = Path(store), host
        self.remote_root, self.max_concurrency = remote_root, max_concurrency

    def _call(self, mode, spec, stdin=None, stdout=None):
        request = {
            "mode": mode,
            "job_id": spec.job_id,
            "spec": asdict(spec),
            "root": self.remote_root,
            "runner": _RUNNER,
            "max_concurrency": self.max_concurrency,
        }
        encoded = base64.b64encode(canonical(request)).decode()
        ssh = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
            self.host,
            "python3 -c " + shlex.quote(_REMOTE) + " " + shlex.quote(encoded),
        ]
        try:
            result = subprocess.run(
                ssh,
                stdin=stdin,
                stdout=stdout or subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TransportError("SSH timed out; remote job may still be running") from exc
        if result.returncode:
            error = result.stderr.decode(errors="replace")[-12000:]
            if result.returncode == 255:
                raise TransportError(error)
            raise RemoteError(error)
        return json.loads(result.stdout) if stdout is None else None

    def stage(self, spec, files):
        if set(files) != set(spec.inputs):
            raise ValueError("Supplied files differ from manifest")
        folder = self.store / spec.job_id
        immutable_json(folder / "spec.json", asdict(spec))
        immutable_json(folder / "remote.json", {"host": self.host, "root": self.remote_root})
        # Archive the exact bytes once; reject input edits after manifest creation.
        with tempfile.TemporaryFile() as archive:
            with tarfile.open(fileobj=archive, mode="w") as tar:
                for name, path in sorted(files.items()):
                    if Path(path).is_symlink() or not Path(path).is_file():
                        raise ValueError("Inputs must be regular files")
                    tar.add(path, arcname=name, recursive=False)
            archive.seek(0)
            result = self._call("stage", spec, stdin=archive)
        return result

    def submit(self, spec):
        folder = self.store / spec.job_id
        immutable_json(folder / "spec.json", asdict(spec))
        immutable_json(folder / "remote.json", {"host": self.host, "root": self.remote_root})
        result = self._call("submit", spec)
        if result["status"] == "busy":
            raise RemoteBusy("Managed GPU concurrency limit reached")
        if result["status"] == "failed":
            raise RemoteError(json.dumps(result))
        return result

    def check(self, spec):
        return self._call("check", spec)

    def retrieve(self, spec):
        destination = self.store / spec.job_id / "artifacts"
        if destination.exists():
            validate_result(destination, spec)
            return destination
        with tempfile.TemporaryFile() as archive:
            self._call("fetch", spec, stdout=archive)
            archive.seek(0)
            publish_archive(archive, destination, spec)
        return destination


def prefect_flow(*, max_polls=120, poll_seconds=5):
    """Create the optional Prefect flow; bounded task retries, no flow retry loop.

    Reinvoke with the same spec after a worker restart. The remote Docker handle
    persists independently from the client and Prefect process.
    """
    try:
        from prefect import flow, task
        from prefect.cache_policies import NO_CACHE
    except ImportError as exc:
        raise ImportError(
            "Run scripts/setup/setup_integrations.py execution and use .venvs/execution/bin/python"
        ) from exc
    if not 0 <= max_polls <= 10000 or not 0 <= poll_seconds <= 60:
        raise ValueError("Invalid bounded polling settings")

    def retry_transient(task, task_run, state):
        return isinstance(state.result(raise_on_failure=False), (RemotePending, TransportError))

    @task(
        retries=3,
        retry_delay_seconds=poll_seconds,
        retry_condition_fn=retry_transient,
        cache_policy=NO_CACHE,
        persist_result=False,
    )
    def stage_and_submit(spec_data, files, options):
        executor = SparkExecutor(**options)
        spec = JobSpec.from_dict(spec_data)
        state = executor.check(spec)
        if state["status"] == "missing":
            executor.stage(spec, files)
        return executor.submit(spec)

    @task(
        retries=max_polls,
        retry_delay_seconds=poll_seconds,
        retry_condition_fn=retry_transient,
        cache_policy=NO_CACHE,
        persist_result=False,
    )
    def await_completion(spec_data, options):
        result = SparkExecutor(**options).check(JobSpec.from_dict(spec_data))
        if result["status"] in ("running", "created", "restarting", "paused"):
            raise RemotePending("Remote container is still " + result["status"])
        if result["status"] != "completed":
            raise RemoteError(json.dumps(result))
        return result

    @task(
        retries=3,
        retry_delay_seconds=poll_seconds,
        retry_condition_fn=retry_transient,
        cache_policy=NO_CACHE,
        persist_result=False,
    )
    def retrieve(spec_data, options):
        return str(SparkExecutor(**options).retrieve(JobSpec.from_dict(spec_data)))

    @flow(name="soccerviz-spark-job", retries=0, persist_result=False)
    def run(spec_data, files, options):
        stage_and_submit(spec_data, files, options)
        await_completion(spec_data, options)
        return retrieve(spec_data, options)

    return run
