"""Checkpoint lifecycle and NNX state restoration integration tests."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import jax.numpy as jnp
import optax
import pytest
from flax import nnx
from jax import Array

from alpacax.trainer.checkpoints import (
    TrainerCheckpointer,
    checkpoint_payload,
    restore_training_state,
)


def _ones(shape: tuple[int, ...]) -> Array:
    return cast(Array, jnp.ones(shape))  # pyright: ignore[reportUnknownMemberType]


def _array(value: Array) -> Array:
    return cast(Array, jnp.array(value))  # pyright: ignore[reportUnknownMemberType]


def make_model_and_optimizer() -> tuple[nnx.Linear, nnx.Optimizer[nnx.Linear]]:
    model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
    return model, optimizer


def train_once(model: nnx.Linear, optimizer: nnx.Optimizer[nnx.Linear]) -> None:
    def loss(m: nnx.Linear) -> Array:
        return jnp.sum(m(_ones((1, 2))))

    grads = nnx.grad(loss)(model)
    optimizer.update(model, grads)


def test_checkpoint_round_trip_restores_nnx_state_and_step(tmp_path: Path) -> None:
    model, optimizer = make_model_and_optimizer()
    train_once(model, optimizer)

    saved_kernel = _array(model.kernel[...])
    saved_optimizer_step = int(cast(nnx.Variable, optimizer.step).value)

    with TrainerCheckpointer(tmp_path) as checkpointer:
        checkpointer.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                metadata={"step": 7},
            ),
            step=7,
        )
        checkpointer.wait_until_finished()
        assert checkpointer.latest() == tmp_path / "7"

    train_once(model, optimizer)
    assert not bool(jnp.array_equal(model.kernel[...], saved_kernel))

    with TrainerCheckpointer(tmp_path) as checkpointer:
        resumed_step = restore_training_state(
            checkpointer,
            model=model,
            optimizer=optimizer,
        )

    assert resumed_step == 7
    assert bool(jnp.array_equal(model.kernel[...], saved_kernel))
    assert int(cast(nnx.Variable, optimizer.step).value) == saved_optimizer_step


def test_checkpointer_closes_on_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpointer = TrainerCheckpointer(tmp_path)
    actual_close = checkpointer.manager.close
    called: list[bool] = []

    def close_and_record() -> None:
        called.append(True)
        actual_close()

    monkeypatch.setattr(checkpointer.manager, "close", close_and_record)

    with pytest.raises(RuntimeError, match="training failed"):
        with checkpointer:
            raise RuntimeError("training failed")

    assert called == [True]


def test_restore_requires_a_checkpoint(tmp_path) -> None:
    model, optimizer = make_model_and_optimizer()

    with TrainerCheckpointer(tmp_path) as checkpointer:
        with pytest.raises(FileNotFoundError, match="No checkpoint"):
            restore_training_state(
                checkpointer,
                model=model,
                optimizer=optimizer,
            )
