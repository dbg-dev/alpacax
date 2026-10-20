from __future__ import annotations

from numbers import Integral
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from flax.nnx import (
    Module,
    Optimizer,
    state,  # pyright: ignore[reportUnknownVariableType]
    update,  # pyright: ignore[reportUnknownVariableType]
)
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, args

from .contracts import Checkpointer, CheckpointPayload


class TrainerCheckpointer:
    """Thin Orbax wrapper for NNX model/optimizer checkpoints."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_to_keep: int = 3,
    ) -> None:
        self.root: Path = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)

        self.manager: CheckpointManager = CheckpointManager(
            self.root,
            options=CheckpointManagerOptions(
                max_to_keep=max_to_keep,
                create=True,
            ),
        )

    def save(
        self,
        state: CheckpointPayload,
        *,
        step: int,
    ) -> Path:
        self.manager.save(
            step,
            args=args.StandardSave(state),
        )
        return self.root / str(step)

    def restore(
        self,
        target: CheckpointPayload,
        path: Path | None = None,
    ) -> CheckpointPayload:
        """Restore into a target PyTree defining NNX types and array sharding."""
        step = self._step_from_path(path) if path is not None else self.manager.latest_step()

        if step is None:
            raise FileNotFoundError(f"No checkpoint found in {self.root}")

        restored = self.manager.restore(
            step,
            args=args.StandardRestore(target),
        )
        return dict(restored)

    def latest(self) -> Path | None:
        step = self.manager.latest_step()
        if step is None:
            return None
        return self.root / str(step)

    def exists(self) -> bool:
        return self.manager.latest_step() is not None

    def wait_until_finished(self) -> None:
        """Block until outstanding asynchronous checkpoint writes complete."""
        self.manager.wait_until_finished()

    def close(self) -> None:
        """Wait for writes and release Orbax resources."""
        self.manager.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @staticmethod
    def _step_from_path(path: Path) -> int:
        try:
            return int(path.name)
        except ValueError as exc:
            raise ValueError(f"Checkpoint path must end with numeric step: {path}") from exc


def checkpoint_payload[ModelT: Module](
    *,
    model: ModelT,
    optimizer: Optimizer[ModelT],
    metadata: dict[str, Any] | None = None,
) -> CheckpointPayload:
    return {
        "model": state(model),
        "optimizer": state(optimizer),
        "metadata": metadata or {},
    }


def restore_training_state[ModelT: Module](
    checkpointer: Checkpointer[CheckpointPayload],
    *,
    model: ModelT,
    optimizer: Optimizer[ModelT],
    path: Path | None = None,
) -> int:
    """Restore a compatible NNX model and optimizer in place.

    All hosts must participate in a distributed restore. The caller owns the
    checkpointer lifecycle and must provide identically structured objects with
    the desired JAX sharding. Returns the last completed training step.
    """
    target = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        metadata={"step": 0},
    )
    restored = checkpointer.restore(target, path)
    completed_step = restored["metadata"]["step"]
    if (
        isinstance(completed_step, bool)
        or not isinstance(completed_step, Integral)
        or completed_step < 0
    ):
        raise ValueError("Checkpoint metadata must contain a non-negative integer 'step'")

    update(model, restored["model"])
    update(optimizer, restored["optimizer"])
    return int(completed_step)
