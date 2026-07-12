"""Replay-time prompt injection for MagenticOne via ``adapter.middleware.patch_with_middlewares``.

Mirrors MetaGPT ``ThinkMiddleware`` + ``AttackMonitor.inject_content`` semantics: when
the replay monitor has reached ``suggestion['step_id']`` (see below) and injection has
not yet run, rewrite the first ``SystemMessage`` in ``create`` / ``create_stream`` with
``REPLAY_PROMPT``.

``monitor.step`` advances on **every** ``run_stream`` event (``core.py`` calls
``record_step`` per event). A single LLM ``create`` can emit **multiple** events, so
``step`` may jump (e.g. 5→8) between two ``create`` calls and never equal ``step_id``
exactly. We therefore inject on the **first** ``create`` / ``create_stream`` where
``monitor.step >= step_id`` (once per run). If ``step > step_id`` at that moment, a
warning is logged (leapfrog / missed exact boundary).

**When ``apply_replay_middlewares_to_model_client`` runs:** from
``MagenticOneAdapter._run_team`` after ``_build_model_client()``, only if ``monitor`` is
an ``AttackMonitor`` (replay with suggestion). Not during ``attack_analysis`` (monitor
is ``None``).

**When ``before`` runs:** every ``create`` / ``create_stream``; it rewrites messages when
``monitor.step >= _attack_step`` (first matching call) and not yet ``_injected``; then
``logger.info`` records the change.
"""

from __future__ import annotations

from typing import Any

from adapter.middleware import Middleware, patch_with_middlewares
from monitor.attack_monitor import AttackMonitor
from monitor.base_monitor import BaseMonitor
from utils.logging import logger
from utils.prompts import REPLAY_PROMPT


class MagenticReplayMiddleware(Middleware):
    """Mutate model ``messages`` before ``ChatCompletionClient.create`` / ``create_stream``."""

    def __init__(self, monitor: AttackMonitor) -> None:
        self.monitor = monitor

    def before(self, ctx: Any) -> None:
        mon = self.monitor
        from autogen_core.models import SystemMessage
        messages = ctx.args[0]
        mlist = list(messages)
        if getattr(mon, "_injected", False):
            return None
        cur, target = int(mon.step), int(mon._attack_step)
        if cur < target:
            return None
        # ``cur > target`` is diagnostic only; injection is still single-shot (``_injected`` below).
        if cur > target:
            logger.warning(
                "[MagenticReplayMiddleware] monitor.step=%s already past attack_step=%s "
                "(stream events can advance multiple times per LLM round); injecting on this "
                "create/create_stream anyway.",
                cur,
                target,
            )
        if not ctx.args:
            return None
        injected = REPLAY_PROMPT.format(
            original_task="",
            injection_info=mon._attack_suggestion,
        )
        for i, m in enumerate(mlist):
            if isinstance(m, SystemMessage):
                base = m.content
                new_content = REPLAY_PROMPT.format(
                    original_task=base,
                    injection_info=mon._attack_suggestion,
                )
                mlist[i] = SystemMessage(content=new_content)
                mon._injected = True  # type: ignore[attr-defined]
                ctx.args = (mlist,) + tuple(ctx.args[1:])
                _log_injection_applied(
                    mon,
                    mode="replace_first_system_message",
                    index=i,
                    original_preview=base,
                    new_preview=new_content,
                )
                return None
        mlist.insert(0, SystemMessage(content=injected))
        mon._injected = True  # type: ignore[attr-defined]
        ctx.args = (mlist,) + tuple(ctx.args[1:])
        _log_injection_applied(
            mon,
            mode="insert_system_message_at_0",
            index=None,
            original_preview="(no SystemMessage in request; inserted new)",
            new_preview=injected,
        )
        return None

    def after(self, ctx: Any, result: Any) -> Any:
        return result


def _log_injection_applied(
    mon: AttackMonitor,
    *,
    mode: str,
    index: int | None,
    original_preview: str,
    new_preview: str,
    preview_len: int = 500,
) -> None:
    inj = str(getattr(mon, "_attack_suggestion", "") or "")
    inj_short = inj if len(inj) <= 400 else inj[:400] + "…"
    orig = original_preview or ""
    orig_short = orig if len(orig) <= preview_len else orig[:preview_len] + "…"
    new = new_preview or ""
    new_short = new if len(new) <= preview_len else new[:preview_len] + "…"
    logger.info(
        "[MagenticReplayMiddleware] REPLAY_PROMPT applied: monitor.step=%s attack_step=%s mode=%s "
        "system_index=%s | injection_info (truncated): %s",
        mon.step,
        getattr(mon, "_attack_step", "?"),
        mode,
        index,
        inj_short,
    )
    logger.info(
        "[MagenticReplayMiddleware] first SystemMessage BEFORE (truncated): %s",
        orig_short,
    )
    logger.info(
        "[MagenticReplayMiddleware] first SystemMessage AFTER (truncated): %s",
        new_short,
    )


def apply_replay_middlewares_to_model_client(
    model_client: Any,
    monitor: BaseMonitor | None,
) -> None:
    """Patch ``create`` and ``create_stream`` on this client when ``monitor`` is ``AttackMonitor``.

    Uses the same ``patch_with_middlewares`` helper as MetaGPT. Safe to call once per
    fresh ``model_client`` instance (e.g. each ``_run_team``).
    """
    if not isinstance(monitor, AttackMonitor):
        return
    mw = MagenticReplayMiddleware(monitor)
    client_cls = type(model_client).__name__
    logger.info(
        "[MagenticReplayMiddleware] Patching model client %s id=%s: create + create_stream "
        "(attack_step=%s, inject on first call with monitor.step >= attack_step).",
        client_cls,
        id(model_client),
        getattr(monitor, "_attack_step", "?"),
    )
    patch_with_middlewares(model_client, "create", [mw])
    patch_with_middlewares(model_client, "create_stream", [mw])
