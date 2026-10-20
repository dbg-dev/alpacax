"""Hugging Face/PyTorch -> Flax NNX adapter around the generic mapping engine."""

from __future__ import annotations

from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import Any, Protocol, Self, cast

from flax.nnx import GraphDef, Module, Param, merge, split, state, to_pure_dict
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import Array
from jax.numpy import array, swapaxes
from returns.result import Result, Success

from alpacax.params.mapping import (
    MappingError,
    MappingRule,
    TensorLike,
    format_mapping_errors,
    map_param_dict,
)

type ModelParamKey = tuple[Hashable, ...]
type HFParamKey = str

type ModelParamDict = dict[ModelParamKey, Array]
type ModelStateDict = dict[ModelParamKey, object]


class HFTensor(TensorLike, Protocol):
    """The tensor operations used by this adapter; no PyTorch import required."""

    def transpose(self, dim0: int, dim1: int) -> Self: ...

    def detach(self) -> Self: ...

    def cpu(self) -> Self: ...

    def numpy(self) -> Any: ...


type HFStateDict = Mapping[HFParamKey, HFTensor]


class HFModelLike(Protocol):
    def state_dict(self) -> HFStateDict: ...


def transpose_torch_last_two(x: HFTensor) -> HFTensor:
    return x.transpose(-2, -1)


def transpose_jax_last_two(x: Array) -> Array:
    return swapaxes(x, -2, -1)


def torch_to_jax(tensor: HFTensor) -> Array:
    return array(tensor.detach().cpu().numpy())


def prepare_target(flax_model: Module) -> tuple[GraphDef[Any], ModelStateDict]:
    graphdef, prestate = split(flax_model)
    # NNX state may contain non-array variables. Flax's flattening utilities
    # do not type their keys precisely, so constrain only the flattened keys.
    return graphdef, cast(ModelStateDict, dict(flatten_dict(to_pure_dict(prestate))))


def rebuild_model(graphdef: GraphDef[Any], post: ModelStateDict) -> Module:
    # Preserve both array and non-array state leaves.
    return merge(graphdef, unflatten_dict(post))


def model_mapper(
    src_model: HFModelLike,
    flax_model: Module,
    rules: Sequence[MappingRule[ModelParamKey, HFParamKey]],
    *,
    strict: bool = False,
    allow_unmapped_dst: Collection[ModelParamKey] = (),
    allow_unused_src: Collection[HFParamKey] = (),
) -> Result[Module, list[MappingError[ModelParamKey, HFParamKey]]]:
    """Map HF/PyTorch parameters into a Flax NNX model.

    By default, unmapped array targets keep their initial values. With
    strict=True, every array-valued NNX Param and source key must be
    covered by rules unless explicitly excepted. Non-parameter state such
    as RNG keys/counters and buffers is preserved without requiring rules.

    Some HF models expose aliases for tied weights (e.g. lm_head.weight).
    List them explicitly in allow_unused_src rather than implicitly ignoring
    all additional source entries.
    """

    src = src_model.state_dict()
    graphdef, pre = prepare_target(flax_model)
    # Only JAX arrays are valid parameter-mapping destinations. Preserve all
    # other NNX state when rebuilding the model after mapping.
    array_targets: ModelParamDict = {
        key: value for key, value in pre.items() if isinstance(value, Array)
    }

    # Not every NNX array is a model parameter. In particular, RngState holds
    # JAX array keys and counters. Strict coverage should require weights to
    # be mapped, not RNG/bookkeeping state carried over from the local model.
    # Keep all array destinations eligible for explicitly declared rules to
    # preserve the existing permissive adapter behavior.
    ignored_non_params: set[ModelParamKey] = set()
    if strict:
        param_state = state(flax_model, Param)
        param_keys = set(flatten_dict(to_pure_dict(param_state)))
        ignored_non_params = array_targets.keys() - param_keys

    result = map_param_dict(
        src=src,
        dst=array_targets,
        rules=rules,
        convert=torch_to_jax,
        transpose_last_two=transpose_torch_last_two,
        strict=strict,
        allow_unmapped_dst=set(allow_unmapped_dst) | ignored_non_params,
        allow_unused_src=allow_unused_src,
    )

    return result.map(lambda post: rebuild_model(graphdef, pre | post))


def model_mapper_or_raise(
    src_model: HFModelLike,
    flax_model: Module,
    rules: Sequence[MappingRule[ModelParamKey, HFParamKey]],
    *,
    strict: bool = False,
    allow_unmapped_dst: Collection[ModelParamKey] = (),
    allow_unused_src: Collection[HFParamKey] = (),
) -> Module:
    result = model_mapper(
        src_model,
        flax_model,
        rules,
        strict=strict,
        allow_unmapped_dst=allow_unmapped_dst,
        allow_unused_src=allow_unused_src,
    )

    if isinstance(result, Success):
        return result.unwrap()

    raise ValueError(format_mapping_errors(result.failure()))
