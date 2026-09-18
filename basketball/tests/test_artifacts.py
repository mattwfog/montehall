import pyarrow as pa
import pytest

from montehall_cv.store.artifacts import (
    ArtifactWriter,
    read_stage,
    stage_complete,
)

SCHEMA = pa.schema([pa.field("a", pa.int32()), pa.field("b", pa.string())])


def test_write_flush_parts_and_read_back(tmp_path):
    stage = tmp_path / "stage"
    writer = ArtifactWriter(stage, SCHEMA, flush_rows=3)
    writer.add_many({"a": i, "b": f"row{i}"} for i in range(8))
    assert not stage_complete(stage)
    writer.close()
    assert stage_complete(stage)
    table = read_stage(stage)
    assert table.num_rows == 8
    assert sorted(table.column("a").to_pylist()) == list(range(8))
    parts = list(stage.glob("part-*.parquet"))
    assert len(parts) == 3  # 3+3+2


def test_incomplete_stage_refuses_read(tmp_path):
    stage = tmp_path / "stage"
    writer = ArtifactWriter(stage, SCHEMA, flush_rows=2)
    writer.add({"a": 1, "b": "x"})
    with pytest.raises(FileNotFoundError):
        read_stage(stage)


def test_rows_written_counts_buffered(tmp_path):
    writer = ArtifactWriter(tmp_path / "s", SCHEMA, flush_rows=100)
    writer.add({"a": 1, "b": "x"})
    assert writer.rows_written == 1


def test_no_tmp_files_left_after_close(tmp_path):
    stage = tmp_path / "stage"
    writer = ArtifactWriter(stage, SCHEMA, flush_rows=2)
    writer.add_many({"a": i, "b": "x"} for i in range(5))
    writer.close()
    assert not list(stage.glob("*.tmp"))


def test_empty_stage_closes_readable(tmp_path):
    """A zero-row stage must read back as an empty table, never as the
    poisoned 'complete but has no part files' state (2026-07-08 harvest)."""
    stage = tmp_path / "stage"
    writer = ArtifactWriter(stage, SCHEMA, flush_rows=2)
    writer.close()
    assert stage_complete(stage)
    table = read_stage(stage)
    assert table.num_rows == 0
    assert table.schema.equals(SCHEMA)
