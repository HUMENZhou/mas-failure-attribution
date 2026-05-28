"""Normalize dataset rows from different parquet schemas into one pipeline shape.

The pipeline historically assumed **kodcode**-style columns. GAIA ``metadata.level*.parquet``
uses different names (e.g. ``Question``, ``Final answer``). Call
``normalize_parquet_task_row`` right after ``Dataset.to_list()`` so ``main`` / ``run_coding_task``
always see:

- ``task_id``
- ``question``
- ``reference_solution`` (string ground truth for logs / eval)
- ``test`` (may be empty for non-code tasks)
- ``data_source`` (for output paths; falls back to dataset file stem if absent)

See module docstring in repo docs or grep for ``task[`` / ``task.get`` for all consumers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _strip_str(val: Any) -> str:
    if val is None:
        return ""
    return str(val).strip()


def normalize_parquet_task_row(
    row: Mapping[str, Any],
    *,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a shallow copy of ``row`` with canonical keys filled for this repo's pipeline."""
    out: dict[str, Any] = dict(row)

    q = _strip_str(out.get("question"))
    if not q:
        q = _strip_str(out.get("Question"))
    out["question"] = q

    tid = out.get("task_id")
    if tid is None or _strip_str(tid) == "":
        tid = out.get("question_ID")
    out["task_id"] = _strip_str(tid) if tid is not None else ""

    ref = _strip_str(out.get("reference_solution"))
    if not ref:
        ref = _strip_str(out.get("Final answer"))
    if not ref:
        ref = _strip_str(out.get("ground_truth"))
    out["reference_solution"] = ref

    test_val = out.get("test")
    out["test"] = "" if test_val is None else (test_val if isinstance(test_val, str) else str(test_val))

    ds = _strip_str(out.get("data_source"))
    if not ds and dataset_path is not None:
        ds = Path(str(dataset_path)).stem
    out["data_source"] = ds or "unknown"

    return out
