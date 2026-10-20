"""Regression coverage for Flax NNX state mapped through the HF adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from returns.result import Failure, Success

from alpacax.integrations.huggingface import (
    HFTensor,
    model_mapper,
    model_mapper_or_raise,
    prepare_target,
    rebuild_model,
)
from alpacax.params.mapping import IDENTITY, MappingError, MappingRule


class MixedStateModel(nnx.Module):
    def __init__(self) -> None:
        self.weight = nnx.Param(jnp.array([1.0, 2.0]))
        self.counter = nnx.Variable(7)


def test_prepare_and_rebuild_preserves_non_array_state() -> None:
    model = MixedStateModel()

    graphdef, flattened = prepare_target(model)
    assert flattened[("counter",)] == 7
    assert isinstance(flattened[("weight",)], type(jnp.array([1.0])))

    # A mapped parameter can change without losing unrelated scalar state.
    flattened[("weight",)] = jnp.array([3.0, 4.0])
    restored = rebuild_model(graphdef, flattened)
    assert isinstance(restored, MixedStateModel)
    assert restored.counter.value == 7
    assert jnp.array_equal(restored.weight.value, jnp.array([3.0, 4.0]))
    assert jnp.array_equal(model.weight.value, jnp.array([1.0, 2.0]))


class FakeTensor:
    """Minimal CPU tensor satisfying the HF adapter without importing torch."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def detach(self) -> Self:
        return self

    def cpu(self) -> Self:
        return self

    def numpy(self) -> np.ndarray:
        return self.values

    def transpose(self, dim0: int, dim1: int) -> Self:
        return type(self)(np.swapaxes(self.values, dim0, dim1))


class FakeHFModel:
    def __init__(self, tensors: dict[str, FakeTensor]) -> None:
        self.tensors = tensors

    def state_dict(self) -> Mapping[str, HFTensor]:
        return self.tensors


def test_hf_strict_coverage_skips_non_array_nnx_state() -> None:
    hf = FakeHFModel({"weight": FakeTensor(np.array([3.0, 4.0]))})
    result = model_mapper(
        hf,
        MixedStateModel(),
        [MappingRule(dst=("weight",), src="weight", transform=IDENTITY)],
        strict=True,
    )
    assert isinstance(result, Success)
    mapped = result.unwrap()
    assert isinstance(mapped, MixedStateModel)
    assert jnp.array_equal(mapped.weight.value, jnp.array([3.0, 4.0]))
    assert mapped.counter.value == 7


def test_hf_strict_coverage_reports_unused_tied_weight_alias() -> None:
    hf = FakeHFModel(
        {
            "weight": FakeTensor(np.array([3.0, 4.0])),
            "lm_head.weight": FakeTensor(np.array([3.0, 4.0])),
        }
    )
    rule = MappingRule(dst=("weight",), src="weight", transform=IDENTITY)
    result = model_mapper(hf, MixedStateModel(), [rule], strict=True)
    assert isinstance(result, Failure)
    assert result.failure() == [
        MappingError(kind="unused_src", dst=None, src="lm_head.weight")
    ]

    mapped = model_mapper_or_raise(
        hf,
        MixedStateModel(),
        [rule],
        strict=True,
        allow_unused_src=("lm_head.weight",),
    )
    assert isinstance(mapped, MixedStateModel)
    assert jnp.array_equal(mapped.weight.value, jnp.array([3.0, 4.0]))


def test_hf_strict_coverage_reports_unmapped_array_target() -> None:
    class ExtraParameterModel(nnx.Module):
        def __init__(self) -> None:
            self.weight = nnx.Param(jnp.array([1.0, 2.0]))
            self.random_init = nnx.Param(jnp.array([9.0]))

    hf = FakeHFModel({"weight": FakeTensor(np.array([3.0, 4.0]))})
    original = ExtraParameterModel()
    rule = MappingRule(dst=("weight",), src="weight", transform=IDENTITY)
    result = model_mapper(hf, original, [rule], strict=True)

    assert isinstance(result, Failure)
    assert result.failure() == [
        MappingError(kind="unmapped_dst", dst=("random_init",))
    ]
    assert jnp.array_equal(original.weight.value, jnp.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="unmapped destination parameter"):
        model_mapper_or_raise(hf, original, [rule], strict=True)
