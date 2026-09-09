"""Property tests for TaskIndexRepairFixer (05_ENGINEERING_STANDARDS.md §5).

TaskIndexRepairFixer is report-only (see src/trajlens/repair/
task_index_repair.py's module docstring): it never rewrites task_index
data. The invariant this suite pins is therefore the opposite of what an
auto-fixing repair would guarantee:

  "apply() never modifies the data shard's task_index column, regardless
   of how many dangling references exist or what value they hold, and the
   report it writes lists exactly those dangling references."

(An earlier version of this suite asserted "repair then re-lint clears
SEMANTIC.TASK_INTEGRITY" -- that invariant described the unsound
nearest-integer auto-fix this fixer no longer performs.)
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.fixtures.builders import build_v3_dataset
from trajlens.checks.protocol import CheckContext, Severity
from trajlens.checks.semantic import TASK_INTEGRITY
from trajlens.model import build_canonical_dataset
from trajlens.repair.task_index_repair import TaskIndexRepairFixer
from trajlens.sources.loader import SourceLoader

CTX = CheckContext(deep=False)

NUM_EPISODES = st.integers(min_value=1, max_value=4)
DANGLING_TASK_INDEX = st.integers(min_value=1, max_value=500)
NUM_DANGLING_FRAMES = st.integers(min_value=1, max_value=3)


def _make_dangling(root, num_episodes: int, dangling_value: int, num_dangling_frames: int) -> None:  # type: ignore[no-untyped-def]
    build_v3_dataset(root, num_episodes=num_episodes)
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    old = pq.read_table(data_path)
    ti_col = old.column("task_index").to_pylist()
    n = min(num_dangling_frames, len(ti_col))
    for i in range(n):
        ti_col[i] = dangling_value
    new = old.set_column(
        old.schema.get_field_index("task_index"), "task_index", pa.array(ti_col, type=pa.int64())
    )
    pq.write_table(new, data_path)


@given(
    num_episodes=NUM_EPISODES,
    dangling_value=DANGLING_TASK_INDEX,
    num_dangling_frames=NUM_DANGLING_FRAMES,
)
@settings(max_examples=15, deadline=None)
def test_apply_never_modifies_task_index_regardless_of_dangling_shape(
    tmp_path_factory,  # type: ignore[no-untyped-def]
    num_episodes: int,
    dangling_value: int,
    num_dangling_frames: int,
) -> None:
    """Invariant: for any number of episodes, any dangling value, any count
    of dangling frames -- apply() leaves the data shard's task_index column
    byte-for-byte identical, and the finding (if it fired) still fires
    afterward, because nothing was corrected."""
    root = tmp_path_factory.mktemp("prop-task-index-repair")
    source = root / "source"
    output = root / "repaired"

    _make_dangling(source, num_episodes, dangling_value, num_dangling_frames)

    handle = SourceLoader().resolve(str(source))
    ds_source = build_canonical_dataset(handle)

    pre_result = TASK_INTEGRITY.run(ds_source, CTX)
    fired = pre_result.severity is Severity.FAIL

    fixer = TaskIndexRepairFixer()
    fixer.apply(ds_source, output)

    # The data/ tree, including the task_index column, must be untouched --
    # this is the entire point of the report-only redesign.
    source_data_path = source / "data" / "chunk-000" / "file-000.parquet"
    output_data_path = output / "data" / "chunk-000" / "file-000.parquet"
    assert source_data_path.read_bytes() == output_data_path.read_bytes(), (
        f"apply() modified task_index data (num_episodes={num_episodes}, "
        f"dangling_value={dangling_value}, num_dangling_frames={num_dangling_frames})"
    )

    handle_after = SourceLoader().resolve(str(output))
    ds_after = build_canonical_dataset(handle_after)
    post_result = TASK_INTEGRITY.run(ds_after, CTX)

    if fired:
        assert post_result.severity is Severity.FAIL, (
            "the finding must still fire after apply() -- nothing was corrected "
            f"(num_episodes={num_episodes}, dangling_value={dangling_value})"
        )
        report_path = output / ".trajlens-task-index-report" / "dangling_task_index_report.json"
        assert report_path.is_file()
        report = json.loads(report_path.read_text())
        assert len(report) == min(num_dangling_frames, 4 * num_episodes)
        assert all(entry["task_index"] == dangling_value for entry in report)
    else:
        # dangling_value happened to collide with a defined index (task_index=0
        # is always defined by build_v3_dataset) -- no finding, no report.
        assert not (output / ".trajlens-task-index-report").exists()
