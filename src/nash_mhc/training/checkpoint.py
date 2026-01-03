"""Orbax checkpointing integration for Nash-MHC."""

from __future__ import annotations

import os
from typing import Any, Mapping

import orbax.checkpoint as ocp


class OrbaxCheckpointManager:
    """Wrapper around Orbax CheckpointManager for Equinox modules."""

    def __init__(
        self,
        directory: str,
        max_to_keep: int = 3,
        enable_async: bool = True,
    ):
        self.directory = os.path.abspath(directory)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=True,
        )

        # Simpler checkpointer for compatibility
        self._manager = ocp.CheckpointManager(
            self.directory,
            ocp.PyTreeCheckpointer(),
            options=options,
        )

    def save(self, state: Any, metrics: Mapping[str, Any] | None = None) -> bool:
        """Saves the train state and optional metrics."""
        step = int(state.step)
        # Using the standard PyTree saving pattern
        return self._manager.save(step, state, metrics=metrics)

    def restore(self, state: Any, step: int | None = None) -> Any:
        """Restores the train state from the latest or specific checkpoint."""
        if step is None:
            step = self._manager.latest_step()
            if step is None:
                return state

        return self._manager.restore(step, items=state)

    def wait_until_finished(self):
        """Blocks until all async checkpointing operations are complete."""
        self._manager.wait_until_finished()

    def close(self):
        """Closes the checkpoint manager."""
        self._manager.close()
