"""REPAIR.TASK_INDEX_REPAIR — report-only fixer for SEMANTIC.TASK_INTEGRITY (dangling task_index).

Detection counterpart: src/trajlens/checks/semantic.py _TaskIntegrityCheck.

Ground truth: meta/tasks.parquet (loaded into CanonicalDataset.task_table).
A dangling task_index is a value referenced by a frame data row that is not
a key in task_table.

This fixer never rewrites task_index data. A dangling task_index is a
broken *reference*, not a value with a principled automatic correction:
the only real repair would require knowing which real-world task a frame
actually belongs to, and that association is unrecoverable from a bare
integer alone.

An earlier version of this fixer "repaired" a dangling reference by
reassigning it to whichever defined task_index was numerically nearest
(argmin(|defined_index - N|)). That heuristic is unsound: task_index is a
categorical identifier, not an ordinal scale — index 7 is not "closer in
meaning" to 6 than to 2. Applying it would silently relabel a frame's
ground-truth language instruction to a different, unrelated task, and
SEMANTIC.TASK_INTEGRITY would then report the dataset as clean, with no
other check in the suite able to detect the mislabeling — the mislabeled
data would look *more* correct, not less, exactly the outcome a "repair"
must never produce (06_SECURITY_AND_THREAT_MODEL.md T9).

apply() therefore only ever reports: it copies the source tree unchanged
and writes a report of every dangling reference found to
<output>/.trajlens-task-index-report/dangling_task_index_report.json,
including — purely as advisory information for a human to act on manually,
never applied to the data — the nearest defined task_index by index
distance, when unambiguous. There is no option to auto-apply that
suggestion; mirrors orphan_shard_report.py's report-first design
(08_ROADMAP.md's repair philosophy: a fixer must never guess content it
cannot recover).

ADR-004 requirements satisfied here:
  - Copy-on-write: source is never opened in write mode; output_path
    receives the full copy, with the report written alongside it.
  - Dry-run by default: dry_run() computes the Diff with zero filesystem
    writes.
  - Round-trip: tests in tests/unit/test_task_index_repair.py and
    tests/property/test_task_index_repair_properties.py verify
    report -> apply() -> task_index data is byte-for-byte unchanged, and
    the report lists exactly the dangling references found.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import structlog

from trajlens.errors import RepairError
from trajlens.model.canonical import CanonicalDataset
from trajlens.repair.protocol import Diff, FeatureFieldChange, RepairSummary, replace_output_dir
from trajlens.sources.paths import safe_join

log = structlog.get_logger(__name__)

FIXER_ID = "REPAIR.TASK_INDEX_REPAIR"
CHECK_ID = "SEMANTIC.TASK_INTEGRITY"

# v3.0 data shards live under data/chunk-*/file-*.parquet — same glob
# timestamp_dedrift.py uses, grounded in the same live lerobot 0.5.2 writer
# and fixture builders (tests/fixtures/builders.py).
_DATA_SHARD_GLOB = "data/chunk-*/file-*.parquet"
_REPORT_DIR = ".trajlens-task-index-report"


@dataclass(frozen=True, slots=True)
class DanglingTaskReference:
    """One frame whose task_index does not resolve in the dataset's task table.

    nearest_defined_task_index_suggestion — advisory only, never written to
    the data. None when no defined task exists at all, or when two or more
    defined indices are equidistant (an ambiguous suggestion is reported as
    no suggestion, not guessed).
    """

    episode_index: int
    frame_index: int
    shard_path: str
    task_index: int
    nearest_defined_task_index_suggestion: int | None


def _dataset_root(ds: CanonicalDataset) -> Path:
    return ds.stats.root


def _nearest_valid_task_suggestion(dangling: int, defined_indices: list[int]) -> int | None:
    """Return the defined task_index nearest to *dangling*, purely as an
    advisory suggestion — never applied to the data (see module docstring
    for why nearest-integer distance is not a principled repair here).

    Returns None if no task is defined at all, or if two or more defined
    indices are equidistant (an ambiguous suggestion is withheld, not guessed).
    """
    if not defined_indices:
        return None
    distances = sorted((abs(d - dangling), d) for d in defined_indices)
    best_distance = distances[0][0]
    candidates = [d for dist, d in distances if dist == best_distance]
    if len(candidates) > 1:
        return None
    return candidates[0]


def find_dangling_task_references(ds: CanonicalDataset) -> list[DanglingTaskReference]:
    """Return every frame whose task_index is not a key in ds.task_table.

    Read-only: scans data shards but never writes anything. Raises
    RepairError only for conditions that make the scan itself undefined
    (wrong format version, no task_index feature declared) — an empty or
    ambiguous task table is not fatal here, since this fixer never needs to
    resolve a definite replacement value; see _nearest_valid_task_suggestion.
    """
    _check_preconditions(ds)

    defined_indices = list(ds.task_table.keys())
    root = _dataset_root(ds)
    references: list[DanglingTaskReference] = []

    # Single glob here — single-writer assumption: source directory must not
    # be mutated by another process between dry_run() and apply().
    shard_paths = sorted(root.glob(_DATA_SHARD_GLOB))
    for shard_abs in shard_paths:
        shard_rel = str(shard_abs.relative_to(root))
        table = pq.read_table(shard_abs)  # type: ignore[no-untyped-call]
        ep_col = table.column("episode_index").to_pylist()
        fi_col = table.column("frame_index").to_pylist()
        ti_col = table.column("task_index").to_pylist()

        for ep, fi, ti in zip(ep_col, fi_col, ti_col, strict=True):
            task_index = int(ti)
            if task_index in ds.task_table:
                continue

            suggestion = _nearest_valid_task_suggestion(task_index, defined_indices)
            references.append(
                DanglingTaskReference(
                    episode_index=int(ep),
                    frame_index=int(fi),
                    shard_path=shard_rel,
                    task_index=task_index,
                    nearest_defined_task_index_suggestion=suggestion,
                )
            )

    return references


class TaskIndexRepairFixer:
    """Detect and report dangling task_index references (ADR-004); never rewrites task_index.

    dry_run() produces a Diff of every frame whose task_index is not a key
    in the dataset's task_table -- a report, not a proposed correction (see
    module docstring for why no correction is principled here).

    apply() copies the source tree unchanged and, if any dangling
    references were found, writes a report to
    <output>/.trajlens-task-index-report/dangling_task_index_report.json.
    task_index data is never modified; changes_written/frames_corrected are
    always 0, matching orphan_shard_report.py's report-only convention.

    Only v3.0 datasets are supported: task_index is a per-frame data-shard
    column only in v3.0 (v2.x resolves tasks per-episode via episodes.jsonl,
    not via a frame-level task_index column). A RepairError is raised for
    v2.x inputs.
    """

    fixer_id: str = FIXER_ID
    check_id: str = CHECK_ID

    def dry_run(self, ds: CanonicalDataset) -> Diff:
        """Compute the dangling-task-reference report without writing anything.

        Returns a Diff with one FeatureFieldChange per dangling reference
        found (feature="episode_<n>_frame_<n>", field="task_index",
        old_value=new_value=the dangling task_index -- a report entry, not
        a correction, mirroring orphan_shard_report.py's convention where
        old_value==new_value signals "detected, not changed"). Returns a
        no-op Diff if every reference already resolves.
        """
        references = find_dangling_task_references(ds)
        changes = tuple(
            FeatureFieldChange(
                feature=f"episode_{r.episode_index}_frame_{r.frame_index}",
                field="task_index",
                old_value=r.task_index,
                new_value=r.task_index,
            )
            for r in references
        )

        diff = Diff(changes=changes, check_id=CHECK_ID, fixer_id=FIXER_ID)
        if diff.is_noop:
            log.info(
                "task_index_repair.dry_run.noop",
                reason="every task_index reference already resolves in the task table",
            )
        else:
            log.info(
                "task_index_repair.dry_run.dangling_found",
                num_dangling=len(references),
                num_shards=len({r.shard_path for r in references}),
            )
        return diff

    def apply(self, ds: CanonicalDataset, output_path: Path) -> RepairSummary:
        """Copy the source tree to *output_path* and write a dangling-task report.

        task_index data is never modified. Without any dangling reference
        found: a plain copy, zero-change RepairSummary. With dangling
        references found: the same plain copy, plus a report file listing
        each one (episode/frame/shard/task_index and, purely as advisory
        information, the nearest defined task_index by index distance when
        unambiguous). *output_path* must not be the source dataset root.
        """
        source_root = _dataset_root(ds)
        if output_path.resolve() == source_root.resolve():
            raise RepairError(
                "output_path must not be the source dataset root (ADR-004 "
                f"copy-on-write). Got: {output_path}"
            )

        references = find_dangling_task_references(ds)

        log.info(
            "task_index_repair.apply.start",
            source=str(source_root),
            output=str(output_path),
            num_dangling=len(references),
        )

        replace_output_dir(output_path)
        shutil.copytree(source_root, output_path)

        if not references:
            log.info("task_index_repair.apply.noop", output=str(output_path))
            return RepairSummary(output_path=output_path, changes_written=0, frames_corrected=0)

        _write_dangling_task_report(output_path, references)

        log.info(
            "task_index_repair.apply.report_written",
            output=str(output_path),
            num_dangling=len(references),
            message=(
                "task_index was NOT modified -- a dangling reference has no "
                "principled automatic correction. See "
                f"{output_path}/{_REPORT_DIR}/dangling_task_index_report.json"
            ),
        )
        # Nothing was corrected: matches orphan_shard_report.py's own
        # report-only convention (changes_written/frames_corrected stay 0
        # even though a report was written -- those fields count data
        # corrected, not bytes of diagnostic output produced).
        return RepairSummary(output_path=output_path, changes_written=0, frames_corrected=0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_preconditions(ds: CanonicalDataset) -> None:
    """Raise RepairError for any condition that makes the scan unsafe or undefined."""
    from trajlens.sources.version import DatasetVersion

    if ds.version is not DatasetVersion.V3_0:
        raise RepairError(
            f"TaskIndexRepairFixer only supports v3.0 datasets; "
            f"got version {ds.version!r}. For v2.x datasets, convert to v3.0 first."
        )
    if "task_index" not in ds.features:
        raise RepairError("TaskIndexRepairFixer requires a 'task_index' feature in info.json.")


def _write_dangling_task_report(output_root: Path, references: list[DanglingTaskReference]) -> None:
    """Write the dangling-task-reference report to output_root/_REPORT_DIR/.

    Never touches task_index data. Every entry names its own advisory
    suggestion explicitly rather than implying trajlens acted on it.
    """
    report_root = safe_join(output_root, _REPORT_DIR)
    report_root.mkdir(parents=True, exist_ok=True)

    manifest = [
        {
            "episode_index": r.episode_index,
            "frame_index": r.frame_index,
            "shard_path": r.shard_path,
            "task_index": r.task_index,
            "nearest_defined_task_index_suggestion": r.nearest_defined_task_index_suggestion,
            "note": (
                "Advisory only. trajlens does not know which real-world task "
                "this frame actually belongs to and has NOT modified "
                "task_index in the data. Review and correct manually."
            ),
        }
        for r in references
    ]
    report_path = report_root / "dangling_task_index_report.json"
    report_path.write_text(json.dumps(manifest, indent=2))
    log.debug("task_index_repair.report_written", path=str(report_path), num_entries=len(manifest))
