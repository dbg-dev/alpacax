from __future__ import annotations

from collections.abc import Iterable

from flax.nnx import Metric, Module, Optimizer
from grain import DataLoader

from .contracts import EvalPhase, HookEntry, LossWithAuxFn, Runner, TrainPhase
from .steps import make_eval_step, make_train_step


def make_train_phase(
    *,
    model: Module,
    optimizer: Optimizer[Module],
    loader: DataLoader,
    metrics: Metric,
    loss_fn: LossWithAuxFn,
    donate: bool = False,
) -> TrainPhase:
    return TrainPhase(
        model=model,
        optimizer=optimizer,
        loader=loader,
        metrics=metrics,
        step=make_train_step(loss_fn, donate=donate),
    )


def make_eval_phase(
    *,
    model: Module,
    loader: DataLoader,
    metrics: Metric,
    loss_fn: LossWithAuxFn,
) -> EvalPhase:
    return EvalPhase(
        model=model,
        loader=loader,
        metrics=metrics,
        step=make_eval_step(loss_fn),
    )


def make_runner(
    *,
    train: TrainPhase,
    eval: EvalPhase | None = None,
    hooks: Iterable[HookEntry] = (),
    max_steps: int | None = None,
) -> Runner:
    return Runner(
        train=train,
        eval=eval,
        hooks=tuple(hooks),
        max_steps=max_steps,
    )
