"""Unit tests for TaskIndexRepairFixer (REPAIR.TASK_INDEX_REPAIR).

TaskIndexRepairFixer is report-only: it never rewrites task_index data.
See src/trajlens/repair/task_index_repair.py's module docstring for why —
an earlier version reassigned a dangling reference to whichever defined
task_index was numerically nearest, which is unsound (task_index is a
categorical identifier, not an ordinal scale) and could silently relabel a
frame's ground-truth language instruction to an unrelated task.

Coverage per 05_ENGINEERING_STANDARDS.md §5 and ADR-004:
  - Happy path: dangling task_index -> apply() -> task_index data is
    byte-for-byte unchanged (not "corrected"), a report is written, and
    SEMANTIC.TASK_INTEGRITY still fires afterward (nothing was fixed).
  - Byte-identity: every file outside the new report directory is untouched,
    including data/ shards (a stronger invariant than the old auto-fix
    version, which legitimately rewrote data/).
  - Report contents: dangling references are listed with an advisory-only
    nearest-task suggestion, explicitly null when no task is defined or the
    suggestion is ambiguous (neither case raises anymore).
  - Dry-run zero-write (mtime pattern).
  - No-op: already-valid dataset -> noop Diff.
  - Failure modes: v2.x dataset rejected, missing task_index feature rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.fixtures.builders import (
    build_v2_dataset,
    build_v3_dataset,
    build_v3_missing_task,
)
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.checks.semantic import TASK_INTEGRITY
from trajlens.errors import RepairError
from trajlens.model import build_canonical_dataset
from trajlens.repair.protocol import Diff, FeatureFieldChange
from trajlens.repair.task_index_repair import (
    CHECK_ID,
    FIXER_ID,
    TaskIndexRepairFixer,
    _nearest_valid_task_suggestion,
    find_dangling_task_references,
)
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(root: Path):  # type: ignore[no-untyped-def]
    handle = SourceLoader().resolve(str(root))
    return build_canonical_dataset(handle)


def _has_task_integrity_finding(root: Path) -> bool:
    ds = _load(root)
    result = TASK_INTEGRITY.run(ds, CTX)
    return result.severity is Severity.FAIL


def _write_empty_tasks_table(root: Path) -> None:
    """Overwrite meta/tasks.parquet with zero rows (no defined tasks at all)."""
    tasks_path = root / "meta" / "tasks.parquet"
    empty = pa.table(
        {"task_index": pa.array([], type=pa.int64()), "task": pa.array([], type=pa.string())}
    )
    pq.write_table(empty, tasks_path)


def _build_two_task_dataset_with_dangling_between(root: Path) -> None:
    """v3.0 dataset with task_index 0 and 10 defined; last frame references 5 (equidistant)."""
    build_v3_dataset(root, num_episodes=1)
    tasks_path = root / "meta" / "tasks.parquet"
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0, 10], type=pa.int64()),
                "task": pa.array(["do thing zero", "do thing ten"]),
            }
        ),
        tasks_path,
    )
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    ti_col = old.column("task_index").to_pylist()
    ti_col[-1] = 5  # equidistant from 0 and 10
    new = old.set_column(
        old.schema.get_field_index("task_index"), "task_index", pa.array(ti_col, type=pa.int64())
    )
    pq.write_table(new, data_path)


def _build_v3_dataset_no_task_index_feature(root: Path, *, camera: str = "top") -> None:
    """v3.0 dataset whose declared features lack 'task_index' (mirrors no_frame_index pattern)."""
    build_v3_dataset(root, camera=camera)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    del info["features"]["task_index"]
    info_path.write_text(json.dumps(info))


# ---------------------------------------------------------------------------
# Fixer identity / metadata
# ---------------------------------------------------------------------------


class TestFixerMetadata:
    def test_ids(self) -> None:
        fixer = TaskIndexRepairFixer()
        assert fixer.fixer_id == FIXER_ID
        assert fixer.check_id == CHECK_ID
        assert fixer.fixer_id == "REPAIR.TASK_INDEX_REPAIR"
        assert fixer.check_id == "SEMANTIC.TASK_INTEGRITY"


# ---------------------------------------------------------------------------
# Nearest-valid-task suggestion (pure function, advisory only)
# ---------------------------------------------------------------------------


class TestNearestValidTaskSuggestion:
    def test_picks_strictly_nearest(self) -> None:
        assert _nearest_valid_task_suggestion(7, [0, 5, 10]) == 5

    def test_exact_match_not_reachable_but_returns_self_if_present(self) -> None:
        assert _nearest_valid_task_suggestion(5, [0, 5, 10]) == 5

    def test_equidistant_returns_none_not_a_raise(self) -> None:
        """An ambiguous suggestion is withheld (None), never guessed or raised --
        this is advisory information only, not a value that must be resolved."""
        assert _nearest_valid_task_suggestion(5, [0, 10]) is None

    def test_no_defined_tasks_returns_none(self) -> None:
        assert _nearest_valid_task_suggestion(5, []) is None


# ---------------------------------------------------------------------------
# Report-only behavior (replaces the old auto-fix round-trip)
# ---------------------------------------------------------------------------


class TestReportOnly:
    def test_apply_never_modifies_task_index_data(self, tmp_path: Path) -> None:
        """The data shard containing the dangling reference must be
        byte-for-byte identical after apply() -- this fixer only reports."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_missing_task(source)

        assert _has_task_integrity_finding(source), "fixture must trigger SEMANTIC.TASK_INTEGRITY"

        fixer = TaskIndexRepairFixer()
        ds = _load(source)
        summary = fixer.apply(ds, output)

        assert summary.output_path == output
        assert summary.frames_corrected == 0
        assert summary.changes_written == 0

        data_path = "data/chunk-000/file-000.parquet"
        assert (source / data_path).read_bytes() == (output / data_path).read_bytes()

    def test_finding_persists_after_apply(self, tmp_path: Path) -> None:
        """SEMANTIC.TASK_INTEGRITY must still fire after apply() -- nothing
        was corrected, so the finding must not silently disappear."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_missing_task(source)

        fixer = TaskIndexRepairFixer()
        fixer.apply(_load(source), output)

        assert _has_task_integrity_finding(output), (
            "the finding must persist -- this fixer never corrects task_index"
        )

    def test_apply_writes_report_for_dangling_references(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_missing_task(source)

        fixer = TaskIndexRepairFixer()
        fixer.apply(_load(source), output)

        report_path = output / ".trajlens-task-index-report" / "dangling_task_index_report.json"
        assert report_path.is_file()
        report = json.loads(report_path.read_text())
        assert len(report) == 1
        entry = report[0]
        assert entry["task_index"] == 99
        assert entry["nearest_defined_task_index_suggestion"] == 0  # advisory only
        assert "not modified" in entry["note"].lower() or "NOT modified" in entry["note"]

    def test_dry_run_produces_no_filesystem_writes(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_missing_task(source)

        before = {p: p.stat().st_mtime for p in source.rglob("*") if p.is_file()}

        fixer = TaskIndexRepairFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        after = {p: p.stat().st_mtime for p in source.rglob("*") if p.is_file()}

        assert before == after, "dry_run() must not touch any file on disk"
        assert not diff.is_noop

    def test_dry_run_diff_contents(self, tmp_path: Path) -> None:
        """The Diff is a report: old_value == new_value == the dangling
        task_index, never a proposed replacement value."""
        source = tmp_path / "source"
        build_v3_missing_task(source)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert isinstance(diff, Diff)
        assert diff.check_id == CHECK_ID
        assert diff.fixer_id == FIXER_ID
        assert len(diff.changes) == 1

        change = diff.changes[0]
        assert isinstance(change, FeatureFieldChange)
        assert change.field == "task_index"
        assert change.old_value == 99
        assert change.new_value == 99  # unchanged -- reported, not corrected


# ---------------------------------------------------------------------------
# Byte-identity outside the report directory (stronger than before: data/ is
# now included, since this fixer never rewrites it)
# ---------------------------------------------------------------------------


class TestByteIdentity:
    def test_everything_byte_identical_outside_the_report_dir(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_missing_task(source)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)
        fixer.apply(ds, output)

        source_files = {p.relative_to(source): p for p in source.rglob("*") if p.is_file()}
        output_files = {
            p.relative_to(output): p
            for p in output.rglob("*")
            if p.is_file() and ".trajlens-task-index-report" not in p.parts
        }
        assert set(source_files) == set(output_files), (
            "apply() must not add or remove any file outside the report dir"
        )

        for rel, src_path in source_files.items():
            out_path = output_files[rel]
            assert src_path.read_bytes() == out_path.read_bytes(), (
                f"{rel} differs between source and repaired output"
            )


# ---------------------------------------------------------------------------
# Formerly-refusal cases: now reported with an advisory None, never raised
# ---------------------------------------------------------------------------


class TestAdvisoryOnlyEdgeCases:
    def test_empty_tasks_table_reports_with_no_suggestion(self, tmp_path: Path) -> None:
        """No defined task exists: the reference is still reported; the
        suggestion is None, and this is no longer a fatal condition."""
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_missing_task(source)
        _write_empty_tasks_table(source)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)

        diff = fixer.dry_run(ds)
        assert not diff.is_noop

        fixer.apply(ds, output)  # must not raise
        report = json.loads(
            (output / ".trajlens-task-index-report" / "dangling_task_index_report.json").read_text()
        )
        assert report[0]["nearest_defined_task_index_suggestion"] is None

    def test_ambiguous_equidistant_reports_with_no_suggestion(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        _build_two_task_dataset_with_dangling_between(source)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)

        diff = fixer.dry_run(ds)
        assert not diff.is_noop

        fixer.apply(ds, output)  # must not raise
        report = json.loads(
            (output / ".trajlens-task-index-report" / "dangling_task_index_report.json").read_text()
        )
        assert report[0]["nearest_defined_task_index_suggestion"] is None


# ---------------------------------------------------------------------------
# find_dangling_task_references (pure detection function)
# ---------------------------------------------------------------------------


class TestFindDanglingTaskReferences:
    def test_finds_the_one_dangling_reference(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_missing_task(source)

        refs = find_dangling_task_references(_load(source))
        assert len(refs) == 1
        assert refs[0].task_index == 99
        assert refs[0].nearest_defined_task_index_suggestion == 0

    def test_clean_dataset_finds_none(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_dataset(source, num_episodes=3)

        assert find_dangling_task_references(_load(source)) == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_clean_dataset_is_noop(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_dataset(source, num_episodes=3)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert diff.is_noop
        assert len(diff.changes) == 0

    def test_clean_dataset_apply_is_noop_summary(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_dataset(source, num_episodes=3)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)
        summary = fixer.apply(ds, output)

        assert summary.frames_corrected == 0
        assert summary.changes_written == 0
        assert output.is_dir()
        assert not (output / ".trajlens-task-index-report").exists()

    def test_zero_episodes_is_noop(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        output = tmp_path / "repaired"
        build_v3_dataset(source, num_episodes=0)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)
        diff = fixer.dry_run(ds)

        assert diff.is_noop
        summary = fixer.apply(ds, output)
        assert summary.frames_corrected == 0

    def test_output_must_not_equal_source(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v3_dataset(source, num_episodes=3)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="copy-on-write"):
            fixer.apply(ds, source)


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_v2_dataset_raises_repair_error(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        build_v2_dataset(source, codebase_version="v2.1", num_episodes=3)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match=r"v3\.0"):
            fixer.dry_run(ds)

    def test_no_task_index_feature_raises_repair_error(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        _build_v3_dataset_no_task_index_feature(source)

        fixer = TaskIndexRepairFixer()
        ds = _load(source)

        with pytest.raises(RepairError, match="task_index"):
            fixer.dry_run(ds)
