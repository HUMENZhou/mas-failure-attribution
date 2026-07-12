"""Serialize AutoGen ``run_stream`` events for JSON traces (prefer structured dumps over ``str()``)."""

from __future__ import annotations

from typing import Any


def _jsonable(value: Any, depth: int = 0, max_depth: int = 12) -> Any:
    if depth > max_depth:
        return "<max-depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) < 200_000 else value[:200_000] + "…(truncated)"
    if isinstance(value, bytes):
        return {"_bytes_b64_hint": f"<{len(value)} bytes>"}
    md = getattr(value, "model_dump", None)
    if callable(md):
        try:
            return md(mode="json")
        except TypeError:
            try:
                return _jsonable(md(), depth + 1, max_depth)
            except Exception:
                pass
        except Exception:
            try:
                return _jsonable(md(), depth + 1, max_depth)
            except Exception:
                pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1, max_depth) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, depth + 1, max_depth) for v in value]
    return str(value)[:8000]


def serialize_stream_event(event: Any) -> dict[str, Any]:
    """Build a JSON-friendly dict for one ``run_stream`` item (``AgentEvent``, ``ChatMessage``, ``TaskResult``, …)."""
    record: dict[str, Any] = {
        "type": type(event).__name__,
        "module": getattr(type(event), "__module__", ""),
    }
    model_dump = getattr(event, "model_dump", None)
    if callable(model_dump):
        try:
            record["model_dump"] = model_dump(mode="json")
            return record
        except TypeError:
            try:
                record["model_dump"] = _jsonable(model_dump())
                return record
            except Exception as err:
                record["model_dump_error"] = repr(err)
        except Exception as err:
            record["model_dump_error"] = repr(err)
    for key in (
        "source",
        "name",
        "content",
        "type",
        "stop_reason",
        "models_usage",
        "messages",
        "message",
        "id",
        "metadata",
    ):
        if hasattr(event, key):
            try:
                record[key] = _jsonable(getattr(event, key))
            except Exception as err:
                record[f"{key}_error"] = repr(err)
    if "content" not in record and "model_dump" not in record:
        try:
            record["repr"] = repr(event)[:4000]
        except Exception as err:
            record["repr_error"] = repr(err)
    return record
