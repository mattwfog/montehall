"""Persistent content-keyed cache for paid VLM verdicts.

The perception -> VLM stages (rim location, shot adjudication, possession
verdicts) each spend Anthropic calls that add up on a full game. A stage that
crashes before stamping its _SUCCESS marker re-runs from scratch, so without a
cache every prior call is re-bought. Keying each verdict by a hash of its exact
request (model + prompt + image/trace bytes) lets a re-run reuse the persisted
answer and pay only for calls not yet made -- the repo's fetch-loop rule
(persist each paid result BEFORE the next request begins) applied to the VLM
tier.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def content_key(*parts: bytes | str) -> str:
    """Stable sha256 over the request parts. Bytes and str both accepted; a
    delimiter is written between parts so their concatenation is unambiguous."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode() if isinstance(part, str) else part)
        h.update(b"\x00")
    return h.hexdigest()


class VlmCache:
    """Append-only JSONL map content_key -> JSON verdict, loaded on open.

    Crash-safe: each put is flushed and fsync'd before the caller issues its
    next paid request, so a killed process never loses a verdict it paid for.
    A torn final line from a hard crash mid-write is tolerated on reload.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._mem: dict[str, dict] = {}
        if path.exists():
            with path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # torn last line after a hard crash; skip it
                    if isinstance(rec, dict) and "key" in rec and "value" in rec:
                        self._mem[rec["key"]] = rec["value"]

    def get(self, key: str) -> dict | None:
        return self._mem.get(key)

    def put(self, key: str, value: dict) -> None:
        self._mem[key] = value
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as fh:
            fh.write(json.dumps({"key": key, "value": value}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
