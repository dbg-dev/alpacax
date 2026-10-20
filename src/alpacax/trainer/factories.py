from __future__ import annotations

from collections.abc import Iterable

from flax.nnx import Metric, Module, Optimizer
from jax import Array

from .contracts import (
    AnyEvalPhase,
    AnyTrainPhase,
    EvalPhase,
    HookEntry,
    LossWithAuxFn,
    Runner,
    TrainPhase,
)
from .steps import make_eval_step, make_train_step


def make_train_phase[ModelT: Module](
    *,
    model: ModelT,
    optimizer: Optimizer[ModelT],
    loader: Iterable[tuple[Array, Array]],
    metrics: Metric,
    loss_fn: LossWithAuxFn,
    donate: bool = False,
) -> TrainPhase[ModelT, Optimizer[ModelT], Metric, Array, Array]:
    return TrainPhase(
        model=model,
        optimizer=optimizer,
        loader=loader,
        metrics=metrics,
        step=make_train_step(loss_fn, donate=donate),
    )


def make_eval_phase[ModelT: Module](
    *,
    model: ModelT,
    loader: Iterable[tuple[Array, Array]],
    metrics: Metric,
    loss_fn: LossWithAuxFn,
) -> EvalPhase[ModelT, Metric, Array, Array]:
    return EvalPhase(
        model=model,
        loader=loader,
        metrics=metrics,
        step=make_eval_step(loss_fn),
    )


def make_runner[TrainT: AnyTrainPhase](
    *,
    train: TrainT,
    eval: AnyEvalPhase | None = None,
    hooks: Iterable[HookEntry] = (),
    max_steps: int | None = None,
) -> Runner[TrainT]:
    return Runner(
        train=train,
        eval=eval,
        hooks=tuple(hooks),
        max_steps=max_steps,
    )
