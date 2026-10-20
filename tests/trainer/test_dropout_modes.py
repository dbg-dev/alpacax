"""Dropout train/eval mode handling, exercised with a real NNX model and jitted steps."""

from __future__ import annotations

from typing import cast, override

import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array

from alpacax.trainer.contracts import HookContext, HookEntry
from alpacax.trainer.core import run
from alpacax.trainer.factories import make_eval_phase, make_runner, make_train_phase
from alpacax.trainer.hooks import eval_hook
from alpacax.trainer.losses import cross_entropy_loss


class DropoutNet(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.linear: nnx.Linear = nnx.Linear(4, 4, rngs=rngs)
        self.dropout: nnx.Dropout = nnx.Dropout(0.5, rngs=rngs)

    @override
    def __call__(self, x: Array) -> Array:
        return self.dropout(self.linear(x))


class RecordingLogger:
    def __init__(self) -> None:
        self.losses: list[float] = []

    def log(self, metrics: dict[str, float], step: int) -> None:
        del step
        self.losses.append(metrics["loss"])

    def close(self) -> None:
        pass


def make_metrics() -> nnx.MultiMetric:
    return nnx.MultiMetric(loss=nnx.metrics.Average("loss"))


def make_batches(n: int = 3) -> list[tuple[Array, Array]]:
    x = cast(Array, jnp.ones((8, 4)))  # pyright: ignore[reportUnknownMemberType]
    y = cast(Array, jnp.zeros((8,), dtype=jnp.int32))  # pyright: ignore[reportUnknownMemberType]
    return [(x, y)] * n


def make_phases(model: DropoutNet):
    optimizer = nnx.Optimizer(model, optax.sgd(0.0), wrt=nnx.Param)  # lr=0: weights stay fixed
    train = make_train_phase(
        model=model,
        optimizer=optimizer,
        loader=make_batches(),
        metrics=make_metrics(),
        loss_fn=cross_entropy_loss,
    )
    evaluation = make_eval_phase(
        model=model,
        loader=make_batches(2),
        metrics=make_metrics(),
        loss_fn=cross_entropy_loss,
    )
    return train, evaluation


def test_eval_hook_loss_is_deterministic_and_restores_dropout() -> None:
    model = DropoutNet(nnx.Rngs(0))
    train, evaluation = make_phases(model)
    logger = RecordingLogger()
    hook = eval_hook(logger)
    ctx = HookContext(step=0, train=train, eval=evaluation)

    hook(ctx)
    hook(ctx)

    assert len(logger.losses) == 2
    assert logger.losses[0] == logger.losses[1]
    # Train model is handed back with dropout active.
    assert model.dropout.deterministic is False


def test_run_enables_dropout_even_if_model_starts_in_eval_mode() -> None:
    model = DropoutNet(nnx.Rngs(0))
    model.eval()  # pyright: ignore[reportUnknownMemberType]
    assert model.dropout.deterministic is True

    train, _ = make_phases(model)
    run(make_runner(train=train))

    assert model.dropout.deterministic is False


def test_full_run_with_eval_hook_ends_in_train_mode_and_logs_stable_eval_loss() -> None:
    model = DropoutNet(nnx.Rngs(0))
    train, evaluation = make_phases(model)
    logger = RecordingLogger()

    run(
        make_runner(
            train=train,
            eval=evaluation,
            # lr is 0 in make_phases, so weights never change: eval loss must be
            # identical at every evaluation if (and only if) dropout is off.
            hooks=[HookEntry(eval_hook(logger), every=1, lead_only=False)],
        )
    )

    assert len(logger.losses) == 3
    assert len(set(logger.losses)) == 1
    assert model.dropout.deterministic is False
