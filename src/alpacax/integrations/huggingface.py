"""Hugging Face/PyTorch -> Flax NNX adapter around the generic mapping engine."""

from __future__ import annotations

from collections.abc import Hashable, MutableMapping, Sequence
from typing import Any, Protocol

from flax.nnx import GraphDef, Module, merge, split, state, to_pure_dict
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import Array
from jax.numpy import array, swapaxes
from returns.result import Result, Success
from torch import Tensor

from alpacax.params.mapping import (
    MappingError,
    MappingRule,
    format_mapping_errors,
    map_param_dict,
)

type ModelParamKey = tuple[Hashable, ...]
type HFParamKey = str

type HFStateDict = MutableMapping[HFParamKey, Tensor]
type ModelParamDict = dict[ModelParamKey, Array]


class HFModelLike(Protocol):
    def state_dict(self) -> MutableMapping[str, Tensor]: ...


def transpose_torch_last_two(x: Tensor) -> Tensor:
    return x.transpose(-2, -1)


def transpose_jax_last_two(x: Array) -> Array:
    return swapaxes(x, -2, -1)


def torch_to_jax(tensor: Tensor) -> Array:
    return array(tensor.detach().cpu().numpy())


def prepare_target(flax_model: Module) -> tuple[GraphDef[Any], ModelParamDict]:
    graphdef, prestate = split(flax_model)
    return graphdef, dict(flatten_dict(to_pure_dict(prestate)))


def rebuild_model(graphdef: GraphDef[Any], post: ModelParamDict) -> Module:
    return merge(graphdef, state(unflatten_dict(post)))


def model_mapper(
    src_model: HFModelLike,
    flax_model: Module,
    rules: Sequence[MappingRule[ModelParamKey, HFParamKey]],
) -> Result[Module, list[MappingError[ModelParamKey, HFParamKey]]]:
    """Map HF/PyTorch parameters into a Flax NNX model.

    The mapping itself is handled by the generic dict-to-dict engine. This
    function only adapts the model-native structures to and from flat dicts.
    """

    src = src_model.state_dict()
    graphdef, pre = prepare_target(flax_model)

    result = map_param_dict(
        src=src,
        dst=pre,
        rules=rules,
        convert=torch_to_jax,
        transpose_last_two=transpose_torch_last_two,
    )

    return result.map(lambda post: rebuild_model(graphdef, post))


def model_mapper_or_raise(
    src_model: HFModelLike,
    flax_model: Module,
    rules: Sequence[MappingRule[ModelParamKey, HFParamKey]],
) -> Module:
    result = model_mapper(src_model, flax_model, rules)

    if isinstance(result, Success):
        return result.unwrap()

    raise ValueError(format_mapping_errors(result.failure()))
