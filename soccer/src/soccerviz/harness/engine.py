"""Durable, single-host specialist execution with immutable artifacts and revisions."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import sqlite3
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Specialist:
    name: str
    version: str
    requires: tuple[str, ...]
    output_schema: str
    resources: str
    evaluate: str
    run: Callable
    uses_reviews: bool = False
    code_dependencies: tuple[Callable, ...] = ()
    libraries: tuple[str, ...] = ()
    optional_requires: tuple[tuple[str, str], ...] = ()
    consumers: tuple[str, ...] = ()
    available: bool = True
    unavailable_reason: str = ""

    def contract(self):
        # Include helpers in the implementation module, not just the entry function.
        source = inspect.getsource(inspect.getmodule(self.run))
        return {
            "name": self.name,
            "version": self.version,
            "requires": self.requires,
            "optional_requires": self.optional_requires,
            "consumers": self.consumers,
            "output_schema": self.output_schema,
            "resources": self.resources,
            "evaluation": self.evaluate,
            "implementation_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "dependency_sha256": {
                f"{fn.__module__}.{fn.__name__}": hashlib.sha256(
                    inspect.getsource(inspect.getmodule(fn)).encode()
                ).hexdigest()
                for fn in self.code_dependencies
            },
            "libraries": {name: importlib.metadata.version(name) for name in self.libraries},
        }


class Harness:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "harness.sqlite"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS objects (
                    id TEXT PRIMARY KEY, body TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, input_id TEXT NOT NULL REFERENCES objects(id),
                    created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS revisions (
                    run_id TEXT NOT NULL REFERENCES runs(id), revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL, review TEXT,
                    PRIMARY KEY(run_id, revision));
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
                    revision INTEGER NOT NULL, stage TEXT NOT NULL, cache_key TEXT NOT NULL,
                    status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
                    output_id TEXT REFERENCES objects(id), error TEXT);
                CREATE TABLE IF NOT EXISTS results (
                    run_id TEXT NOT NULL, revision INTEGER NOT NULL, stage TEXT NOT NULL,
                    cache_key TEXT NOT NULL, output_id TEXT NOT NULL REFERENCES objects(id),
                    PRIMARY KEY(run_id, revision, stage));
                CREATE TABLE IF NOT EXISTS colony_manifests (
                    run_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    manifest_id TEXT NOT NULL REFERENCES objects(id), created_at TEXT NOT NULL,
                    PRIMARY KEY(run_id,revision,manifest_id),
                    FOREIGN KEY(run_id,revision) REFERENCES revisions(run_id,revision));
                CREATE TABLE IF NOT EXISTS review_imports (
                    run_id TEXT NOT NULL REFERENCES runs(id), source_key TEXT NOT NULL,
                    payload_id TEXT NOT NULL REFERENCES objects(id), receipt TEXT NOT NULL,
                    PRIMARY KEY(run_id, source_key));
                CREATE TABLE IF NOT EXISTS attachments (
                    run_id TEXT NOT NULL, revision INTEGER NOT NULL, kind TEXT NOT NULL,
                    object_id TEXT NOT NULL REFERENCES objects(id), created_at TEXT NOT NULL,
                    PRIMARY KEY(run_id, revision, kind, object_id),
                    FOREIGN KEY(run_id,revision) REFERENCES revisions(run_id,revision));
                CREATE TRIGGER IF NOT EXISTS immutable_object_update BEFORE UPDATE ON objects
                    BEGIN SELECT RAISE(ABORT, 'Artifacts are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_object_delete BEFORE DELETE ON objects
                    BEGIN SELECT RAISE(ABORT, 'Artifacts are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_revision_update BEFORE UPDATE ON revisions
                    BEGIN SELECT RAISE(ABORT, 'Reviews are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS immutable_revision_delete BEFORE DELETE ON revisions
                    BEGIN SELECT RAISE(ABORT, 'Reviews are append-only'); END;
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def execution_lock(self):
        # Process-scoped lock releases on SIGKILL; no stale lease timeout is necessary.
        with (self.root / "execution.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "Another specialist execution is active in this store"
                ) from error
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def put(db, body):
        key = digest(body)
        db.execute("INSERT OR IGNORE INTO objects VALUES (?, ?, ?)", (key, canonical(body), now()))
        return key

    def get(self, key):
        with self.connect() as db:
            row = db.execute("SELECT body FROM objects WHERE id=?", (key,)).fetchone()
        if row is None:
            raise ValueError(f"Missing artifact {key}")
        value = json.loads(row[0])
        if digest(value) != key:
            raise ValueError(f"Artifact integrity check failed: {key}")
        return value

    def create(self, evidence):
        run = uuid.uuid4().hex
        with self.connect() as db:
            key = self.put(db, evidence)
            db.execute("INSERT INTO runs VALUES (?, ?, ?)", (run, key, now()))
            db.execute("INSERT INTO revisions VALUES (?, 0, ?, NULL)", (run, now()))
        return run

    def context(self, run, revision=None):
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run,)).fetchone()
            revisions = db.execute(
                "SELECT * FROM revisions WHERE run_id=? ORDER BY revision", (run,)
            ).fetchall()
        if row is None:
            raise ValueError("Unknown run")
        revision = revisions[-1]["revision"] if revision is None else revision
        selected = [r for r in revisions if r["revision"] <= revision]
        if not selected or selected[-1]["revision"] != revision:
            raise ValueError("Unknown revision")
        return {
            "run_id": run,
            "revision": revision,
            "input_id": row["input_id"],
            "known_at": selected[-1]["created_at"],
            "reviews": [
                dict(r) | {"review": json.loads(r["review"])} for r in selected if r["review"]
            ],
        }

    def review(self, run, review):
        return self.review_batch(run, [review], source_key=uuid.uuid4().hex)["revision"]

    def review_batch(self, run, reviews, *, source_key, expected_revision=None):
        """Validate then atomically append an import; a retry cannot duplicate its revisions."""
        from soccerviz.vision.specialists import validate_review

        if not isinstance(source_key, str) or not source_key.strip():
            raise ValueError("A stable import source key is required")
        if not isinstance(reviews, list) or not reviews:
            raise ValueError("At least one reviewed correction is required")
        context = self.context(run)
        evidence = self.get(context["input_id"])
        for review in reviews:
            validate_review(review, evidence)
        payload = {"reviews": reviews}
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT * FROM review_imports WHERE run_id=? AND source_key=?", (run, source_key)
            ).fetchone()
            if prior:
                if prior["payload_id"] != digest(payload):
                    raise ValueError("Import source key was already used for different corrections")
                return json.loads(prior["receipt"]) | {"reused": True}
            latest = db.execute(
                "SELECT MAX(revision) FROM revisions WHERE run_id=?", (run,)
            ).fetchone()[0]
            if expected_revision is not None and latest != expected_revision:
                raise ValueError(
                    "Run has newer reviews; reload the latest revision before applying"
                )
            for offset, review in enumerate(reviews, 1):
                db.execute(
                    "INSERT INTO revisions VALUES (?, ?, ?, ?)",
                    (run, latest + offset, now(), canonical(review)),
                )
            receipt = {
                "run_id": run,
                "source_key": source_key,
                "count": len(reviews),
                "previous_revision": latest,
                "revision": latest + len(reviews),
                "reused": False,
            }
            db.execute(
                "INSERT INTO review_imports VALUES (?, ?, ?, ?)",
                (run, source_key, self.put(db, payload), canonical(receipt)),
            )
        return receipt

    def attach(self, run, kind, payload, revision=None):
        context = self.context(run, revision)
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("Attachment kind is required")
        with self.connect() as db:
            key = self.put(db, payload)
            db.execute(
                "INSERT OR IGNORE INTO attachments VALUES (?, ?, ?, ?, ?)",
                (run, context["revision"], kind, key, now()),
            )
        return key

    def list_runs(self):
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT r.id, r.created_at, MAX(v.revision) AS revision FROM runs r "
                    "JOIN revisions v ON v.run_id=r.id GROUP BY r.id ORDER BY r.created_at DESC"
                )
            ]

    def execute(self, run, specialists, revision=None, max_stages=None):
        if max_stages is not None and max_stages < 1:
            raise ValueError("Stage budget must be positive")
        from soccerviz.harness.colony import ColonyRegistry

        registry = (
            specialists if isinstance(specialists, ColonyRegistry) else ColonyRegistry(specialists)
        )
        ordered = registry.execution_order()
        manifest = registry.manifest()
        with self.execution_lock():
            context = self.context(run, revision)
            revision = context["revision"]
            with self.connect() as db:
                manifest_id = self.put(db, manifest)
                db.execute(
                    "INSERT OR IGNORE INTO colony_manifests VALUES (?,?,?,?)",
                    (run, revision, manifest_id, now()),
                )
                db.execute(
                    "UPDATE attempts SET status='interrupted', finished_at=? "
                    "WHERE status='running'",
                    (now(),),
                )
                # Rebuild the current view so a failed recomputation cannot expose a stale brief.
                db.execute("DELETE FROM results WHERE run_id=? AND revision=?", (run, revision))
            outputs, decisions, executed = {}, [], 0
            for spec in ordered:
                optional = {
                    name: {
                        "status": "present" if name in outputs else "absent",
                        "artifact": outputs.get(name),
                        "absent_behavior": behavior,
                    }
                    for name, behavior in spec.optional_requires
                }
                parents = {name: outputs[name] for name in spec.requires}
                parents.update(
                    {
                        name: value["artifact"]
                        for name, value in optional.items()
                        if value["status"] == "present"
                    }
                )
                signature = {
                    "contract": spec.contract(),
                    "input": context["input_id"],
                    "parents": parents,
                    "optional_inputs": optional,
                    "reviews": context["reviews"] if spec.uses_reviews else [],
                }
                key = digest(signature)
                with self.connect() as db:
                    cached = db.execute(
                        "SELECT output_id FROM attempts WHERE run_id=? AND cache_key=? "
                        "AND status='completed' ORDER BY id DESC LIMIT 1",
                        (run, key),
                    ).fetchone()
                if cached:
                    output = cached[0]
                    self.get(output)
                    action = "reused"
                else:
                    if max_stages is not None and executed >= max_stages:
                        return {
                            "run_id": run,
                            "revision": revision,
                            "status": "paused",
                            "stages": decisions,
                            "next_stage": spec.name,
                            "colony_manifest_id": manifest_id,
                            "scientific_status": "unmeasured",
                        }
                    with self.connect() as db:
                        attempt = db.execute(
                            "INSERT INTO attempts(run_id,revision,stage,cache_key,status,started_at) "
                            "VALUES (?,?,?,?, 'running',?)",
                            (run, revision, spec.name, key, now()),
                        ).lastrowid
                    try:
                        payload = spec.run(
                            self.get(context["input_id"]),
                            {
                                **{
                                    name: self.get(value)["payload"]
                                    for name, value in parents.items()
                                },
                                **{
                                    name: None
                                    for name, value in optional.items()
                                    if value["status"] == "absent"
                                },
                            },
                            context["reviews"] if spec.uses_reviews else [],
                        )
                        if (
                            not isinstance(payload, dict)
                            or payload.get("schema") != spec.output_schema
                        ):
                            raise ValueError(f"Invalid output schema from {spec.name}")
                        envelope = {"signature": signature, "payload": payload}
                        with self.connect() as db:
                            output = self.put(db, envelope)
                            db.execute(
                                "UPDATE attempts SET status='completed',output_id=?,"
                                "finished_at=? WHERE id=?",
                                (output, now(), attempt),
                            )
                    except BaseException as error:
                        with self.connect() as db:
                            db.execute(
                                "UPDATE attempts SET status='failed',error=?,finished_at=? "
                                "WHERE id=?",
                                (str(error), now(), attempt),
                            )
                        raise
                    executed += 1
                    action = "executed"
                with self.connect() as db:
                    db.execute(
                        "INSERT INTO results VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(run_id,revision,stage) DO UPDATE SET "
                        "cache_key=excluded.cache_key,output_id=excluded.output_id",
                        (run, revision, spec.name, key, output),
                    )
                outputs[spec.name] = output
                decisions.append({"stage": spec.name, "action": action, "artifact": output})
            return {
                "run_id": run,
                "revision": revision,
                "status": "completed",
                "stages": decisions,
                "scientific_status": "unmeasured",
                "colony_manifest_id": manifest_id,
                "unavailable": manifest["unavailable"],
            }

    def status(self, run, revision=None):
        context = self.context(run, revision)
        with self.connect() as db:
            results = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM results WHERE run_id=? AND revision=?",
                    (run, context["revision"]),
                )
            ]
            attempts = [
                dict(r)
                for r in db.execute("SELECT * FROM attempts WHERE run_id=? ORDER BY id", (run,))
            ]
            attachments = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM attachments WHERE run_id=? AND revision<=? ORDER BY created_at",
                    (run, context["revision"]),
                )
            ]
            manifests = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM colony_manifests WHERE run_id=? AND revision=? ORDER BY created_at,manifest_id",
                    (run, context["revision"]),
                )
            ]
        return context | {
            "colony_manifests": manifests,
            "results": results,
            "attempts": attempts,
            "attachments": attachments,
            "attachment_clock": "Attachments retain separate created_at timestamps; known_at applies to state reviews",
        }

    def export(self, run, revision=None):
        status = self.status(run, revision)
        manifest_id = digest(
            {
                "input": status["input_id"],
                "revision": status["revision"],
                "results": status["results"],
                "attachments": status["attachments"],
                "colony_manifests": status["colony_manifests"],
            }
        )[:16]
        target = self.root / "exports" / run / f"revision-{status['revision']}" / manifest_id
        target.mkdir(parents=True, exist_ok=True)
        pending = [
            status["input_id"],
            *[r["output_id"] for r in status["results"]],
            *[r["object_id"] for r in status["attachments"]],
            *[r["manifest_id"] for r in status["colony_manifests"]],
        ]
        seen = set()
        while pending:
            key = pending.pop()
            if key in seen:
                continue
            seen.add(key)
            body = self.get(key)
            (target / f"{key}.json").write_text(json.dumps(body, indent=2) + "\n")
            pending.extend(body.get("signature", {}).get("parents", {}).values())
        for row in status["results"]:
            if row["stage"] == "brief":
                body = self.get(row["output_id"])
                links = "\n".join(
                    f"- [{name}]({key}.json)" for name, key in body["signature"]["parents"].items()
                )
                text = (
                    f"Run `{run}` · revision {status['revision']} · known at {status['known_at']}\n\n"
                    + body["payload"]["markdown"]
                    + "\n\nEvidence artifacts:\n\n"
                    + links
                    + "\n"
                )
                (target / "brief.md").write_text(text)
        (target / "run.json").write_text(json.dumps(status, indent=2) + "\n")
        return target
