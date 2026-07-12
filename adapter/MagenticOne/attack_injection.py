"""MagenticOne replay injection (compat shim).

Injection uses the same ``adapter.middleware.patch_with_middlewares`` pattern as MetaGPT.
See ``MagenticReplayMiddleware`` in ``adapter.MagenticOne.middlewares.replay``.
``MagenticOneAdapter._run_team`` applies it to the model client when ``monitor`` is an
``AttackMonitor``.

``magentic_injection_session`` is a no-op for older scripts that wrapped ``run_coding_task``.
"""

from __future__ import annotations

import contextlib

from monitor.base_monitor import BaseMonitor
from utils.logging import logger


@contextlib.contextmanager
def magentic_injection_session(monitor: BaseMonitor | None):
    """Deprecated no-op: replay middleware is wired in ``adapter.MagenticOne.core``."""
    logger.debug(
        "magentic_injection_session is deprecated; use core._run_team middleware wiring."
    )
    yield
