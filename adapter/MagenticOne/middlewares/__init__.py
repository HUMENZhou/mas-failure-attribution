"""MagenticOne-specific middleware (same hook pattern as ``adapter.middleware``)."""

from adapter.MagenticOne.middlewares.replay import (
    MagenticReplayMiddleware,
    apply_replay_middlewares_to_model_client,
)

__all__ = ["MagenticReplayMiddleware", "apply_replay_middlewares_to_model_client"]
