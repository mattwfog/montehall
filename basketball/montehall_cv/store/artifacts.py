"""Stage artifacts: incremental, resumable parquet datasets.

Every pipeline stage writes its output as it is produced (part files under
<out>/<stage>/), then stamps _SUCCESS on completion. A re-run skips any
stage whose _SUCCESS exists — no work is repeated, no partial output is
trusted. This is the repo's fetch-loop contract applied to GPU stages.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SUCCESS_MARKER = "_SUCCESS"


class ArtifactWriter:
    """Buffered part-file writer for one stage's output dataset."""

    def __init__(self, stage_dir: Path, schema: pa.Schema, flush_rows: int = 50_000) -> None:
        if flush_rows <= 0:
            raise ValueError("flush_rows must be positive")
        self._dir = stage_dir
        self._schema = schema
        self._flush_rows = flush_rows
        self._buffer: list[dict] = []
        self._part = 0
        self._rows_written = 0
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def rows_written(self) -> int:
        return self._rows_written + len(self._buffer)

    def add(self, row: dict) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self._flush_rows:
            self._flush()

    def add_many(self, rows: Iterable[dict]) -> None:
        for row in rows:
            self.add(row)

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self._schema)
        part_path = self._dir / f"part-{self._part:05d}.parquet"
        tmp_path = part_path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp_path, compression="zstd")
        tmp_path.rename(part_path)
        self._rows_written += len(self._buffer)
        self._buffer = []
        self._part += 1

    def close(self) -> None:
        self._flush()
        if self._part == 0:
            # A completed-but-empty stage is a valid dataset, not a poisoned
            # one: stamp a schema-correct empty part so read_stage returns
            # 0 rows instead of "complete but has no part files".
            empty = pa.Table.from_pylist([], schema=self._schema)
            part_path = self._dir / "part-00000.parquet"
            tmp_path = part_path.with_suffix(".parquet.tmp")
            pq.write_table(empty, tmp_path, compression="zstd")
            tmp_path.rename(part_path)
            self._part = 1
        (self._dir / SUCCESS_MARKER).touch()


def stage_complete(stage_dir: Path) -> bool:
    return (stage_dir / SUCCESS_MARKER).exists()


def read_stage(stage_dir: Path) -> pa.Table:
    if not stage_complete(stage_dir):
        raise FileNotFoundError(f"stage incomplete (no {SUCCESS_MARKER}): {stage_dir}")
    parts = sorted(stage_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"stage complete but has no part files: {stage_dir}")
    return pa.concat_tables([pq.read_table(p) for p in parts])
