from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from flax.nnx import Metric, Module, Optimizer
from grain import DataLoader
from jax import Array

type CheckpointPayload = dict[str, Any]
type LossWithAuxFn = Callable[[Module, Array, Array], tuple[Array, Array]]
type TrainStep = Callable[
    [Module, Optimizer[Module], Metric, Array, Array],
    None,
]
type EvalStep = Callable[[Module, Metric, Array, Array], None]


@runtime_checkable
class Checkpointer[StateT](Protocol):
    """Persistence boundary for training state."""

    def save(self, state: StateT, *, step: int) -> Path:
        """Persist state and return the checkpoint path."""
        ...

    def restore(self, path: Path | None = None) -> StateT:
        """Restore from a specific checkpoint, or latest if omitted."""
        ...

    def latest(self) -> Path | None:
        """Return the latest checkpoint path, if one exists."""
        ...

    def exists(self) -> bool:
        """Return whether a checkpoint is available."""
        ...

    def close(self) -> None:
        """Release any resources held by the checkpointer."""
        ...


class MetricsLogger(Protocol):
    def log(self, metrics: dict[str, float], step: int) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class TrainPhase:
    model: Module
    optimizer: Optimizer[Module]
    loader: DataLoader
    metrics: Metric
    step: TrainStep


@dataclass(slots=True)
class EvalPhase:
    model: Module
    loader: DataLoader
    metrics: Metric
    step: EvalStep


@dataclass(frozen=True, slots=True)
class HookContext:
    step: int
    train: TrainPhase
    eval: EvalPhase | None = None


type Hook = Callable[[HookContext], None]


@dataclass(frozen=True, slots=True)
class HookEntry:
    fn: Hook
    every: int = 1
    lead_only: bool = True
    on_final: bool = True

    def __post_init__(self) -> None:
        if self.every <= 0:
            raise ValueError("hook interval 'every' must be positive")

    def should_run(self, step: int) -> bool:
        """Whether a periodic hook is due after this many completed updates."""
        return step > 0 and step % self.every == 0


@dataclass(slots=True)
class Runner:
    train: TrainPhase
    eval: EvalPhase | None = None
    hooks: tuple[HookEntry, ...] = ()
    max_steps: int | None = None
