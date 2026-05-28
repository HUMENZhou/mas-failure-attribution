"""Recover strategist JSON from Magentic ``exp_out/magentic_trace.json`` when agents omit file writes."""

from __future__ import annotations

import json
from json import JSONDecoder
from pathlib import Path
from typing import Any

from utils.common import read_json_file

_ATTACK_KEYS = frozenset(
    {"step_id", "fault_code", "attacked_content", "mistake_reason", "related_error"}
)
_DIAGNOSE_KEYS = frozenset(
    {"step_id", "fault_code", "suggested_fix", "mistake_reason", "related_error"}
)


def _strip_markdown_json_fence(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _iter_json_objects_in_text(s: str):
    dec = JSONDecoder()
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n or s[i] != "{":
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(s, i)
            yield obj
            i = end
        except json.JSONDecodeError:
            i += 1


def _collect_trace_json_candidate_strings(obj: Any, out: list[str]) -> None:
    if isinstance(obj, str):
        if "fault_code" in obj and "step_id" in obj and len(obj) > 30:
            out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_trace_json_candidate_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_trace_json_candidate_strings(v, out)


def _valid_step_id(sid: Any) -> bool:
    if isinstance(sid, bool) or not isinstance(sid, (int, float)):
        return False
    if isinstance(sid, float) and not sid.is_integer():
        return False
    return True


def _valid_related_error(rel: Any) -> bool:
    if not isinstance(rel, list):
        return False
    for x in rel:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return False
        if isinstance(x, float) and not x.is_integer():
            return False
    return True


def _is_valid_attack_record(o: Any) -> bool:
    if not isinstance(o, dict) or not _ATTACK_KEYS.issubset(o.keys()):
        return False
    if not _valid_step_id(o.get("step_id")):
        return False
    fc = o.get("fault_code")
    if not isinstance(fc, str) or not fc.strip():
        return False
    for k in ("attacked_content", "mistake_reason"):
        v = o.get(k)
        if not isinstance(v, str):
            return False
    return _valid_related_error(o.get("related_error"))


def _is_valid_diagnose_record(o: Any) -> bool:
    if not isinstance(o, dict) or not _DIAGNOSE_KEYS.issubset(o.keys()):
        return False
    if not _valid_step_id(o.get("step_id")):
        return False
    fc = o.get("fault_code")
    if not isinstance(fc, str) or not fc.strip():
        return False
    for k in ("suggested_fix", "mistake_reason"):
        v = o.get(k)
        if not isinstance(v, str):
            return False
    return _valid_related_error(o.get("related_error"))


def _coerce_attack_record(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_id": int(obj["step_id"]),
        "fault_code": str(obj["fault_code"]).strip(),
        "attacked_content": obj["attacked_content"],
        "mistake_reason": obj["mistake_reason"],
        "related_error": [int(x) for x in obj["related_error"]],
    }


def _coerce_diagnose_record(obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_id": int(obj["step_id"]),
        "fault_code": str(obj["fault_code"]).strip(),
        "suggested_fix": obj["suggested_fix"],
        "mistake_reason": obj["mistake_reason"],
        "related_error": [int(x) for x in obj["related_error"]],
    }


def _scan_trace_for_last_match(
    workspace: Path,
    *,
    predicate: Any,
    coerce: Any,
) -> dict[str, Any] | None:
    trace_path = Path(workspace).resolve() / "exp_out" / "magentic_trace.json"
    if not trace_path.is_file():
        return None
    try:
        trace = read_json_file(trace_path)
    except (FileNotFoundError, ValueError, OSError):
        return None
    events = trace.get("events")
    if not isinstance(events, list):
        return None
    last: dict[str, Any] | None = None
    for ev in events:
        texts: list[str] = []
        _collect_trace_json_candidate_strings(ev, texts)
        for raw in texts:
            for chunk in (_strip_markdown_json_fence(raw), raw):
                for obj in _iter_json_objects_in_text(chunk):
                    if predicate(obj):
                        last = coerce(obj)
    return last


def recover_attack_analysis_from_magentic_trace(workspace: Path) -> dict[str, Any] | None:
    """Last valid attack-shaped JSON in trace (``attacked_content`` field)."""
    return _scan_trace_for_last_match(
        workspace, predicate=_is_valid_attack_record, coerce=_coerce_attack_record
    )


def recover_diagnose_analysis_from_magentic_trace(workspace: Path) -> dict[str, Any] | None:
    """Last valid diagnose-shaped JSON in trace (``suggested_fix`` field)."""
    return _scan_trace_for_last_match(
        workspace, predicate=_is_valid_diagnose_record, coerce=_coerce_diagnose_record
    )
