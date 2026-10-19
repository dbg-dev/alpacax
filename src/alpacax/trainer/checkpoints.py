from __future__ import annotations

from pathlib import Path

from flax.nnx import Module, Optimizer, state
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, args

from .contracts import CheckpointPayload


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
        payload: CheckpointPayload,
        *,
        step: int,
    ) -> Path:
        self.manager.save(
            step,
            args=args.StandardSave(payload),
        )
        return self.root / str(step)

    def restore(self, path: Path | None = None) -> CheckpointPayload:
        step = self._step_from_path(path) if path is not None else self.manager.latest_step()

        if step is None:
            raise FileNotFoundError(f"No checkpoint found in {self.root}")

        restored = self.manager.restore(
            step,
            args=args.StandardRestore(),
        )
        return dict(restored)

    def latest(self) -> Path | None:
        step = self.manager.latest_step()
        if step is None:
            return None
        return self.root / str(step)

    def exists(self) -> bool:
        return self.manager.latest_step() is not None

    def close(self) -> None:
        self.manager.close()

    @staticmethod
    def _step_from_path(path: Path) -> int:
        try:
            return int(path.name)
        except ValueError as exc:
            raise ValueError(f"Checkpoint path must end with numeric step: {path}") from exc


def checkpoint_payload(
    *,
    model: Module,
    optimizer: Optimizer[Module],
    metadata: dict[str, Any] | None = None,
) -> CheckpointPayload:
    return {
        "model": state(model),
        "optimizer": state(optimizer),
        "metadata": metadata or {},
    }
