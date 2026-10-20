from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from flax.nnx import Module
from jax import Array

type CheckpointPayload = dict[str, Any]
type LossWithAuxFn = Callable[[Module, Array, Array], tuple[Array, Array]]
type TrainStep[ModelT, OptimizerT, MetricsT, X, Y] = Callable[
    [ModelT, OptimizerT, MetricsT, X, Y], None
]
type EvalStep[ModelT, MetricsT, X, Y] = Callable[[ModelT, MetricsT, X, Y], None]


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


class TrainEvalModel(Protocol):
    """The only model lifecycle operations required by the runner and hooks."""

    def train(self) -> None: ...
    def eval(self) -> None: ...


class MetricsLike(Protocol):
    """Metric operations used by logging and evaluation hooks."""

    def compute(self) -> Mapping[str, object]: ...
    def reset(self) -> None: ...


@dataclass(slots=True)
class TrainPhase[
    ModelT: TrainEvalModel,
    OptimizerT,
    MetricsT: MetricsLike,
    X,
    Y,
]:
    model: ModelT
    optimizer: OptimizerT
    loader: Iterable[tuple[X, Y]]
    metrics: MetricsT
    step: TrainStep[ModelT, OptimizerT, MetricsT, X, Y]


@dataclass(slots=True)
class EvalPhase[ModelT: TrainEvalModel, MetricsT: MetricsLike, X, Y]:
    model: ModelT
    loader: Iterable[tuple[X, Y]]
    metrics: MetricsT
    step: EvalStep[ModelT, MetricsT, X, Y]


# Hooks operate on a shared, type-erased phase boundary. Individual phases
# preserve their concrete model, optimiser, metric and batch types.
type AnyTrainPhase = TrainPhase[Any, Any, Any, Any, Any]
type AnyEvalPhase = EvalPhase[Any, Any, Any, Any]


@dataclass(frozen=True, slots=True)
class HookContext:
    step: int
    train: AnyTrainPhase
    eval: AnyEvalPhase | None = None


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
class Runner[TrainT: AnyTrainPhase]:
    train: TrainT
    eval: AnyEvalPhase | None = None
    hooks: tuple[HookEntry, ...] = ()
    max_steps: int | None = None
