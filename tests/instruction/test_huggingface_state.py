"""Regression coverage for Flax NNX state mapped through the HF adapter."""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from alpacax.integrations.huggingface import prepare_target, rebuild_model


class MixedStateModel(nnx.Module):
    def __init__(self) -> None:
        self.weight = nnx.Param(jnp.array([1.0, 2.0]))
        self.counter = nnx.Variable(7)


def test_prepare_and_rebuild_preserves_non_array_state() -> None:
    model = MixedStateModel()

    graphdef, flattened = prepare_target(model)
    assert flattened[("counter",)] == 7
    assert isinstance(flattened[("weight",)], type(jnp.array([1.0])))

    restored = rebuild_model(graphdef, flattened)
    assert restored.counter.value == 7
    assert jnp.array_equal(restored.weight.value, model.weight.value)
