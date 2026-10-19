"""Generic parameter-dict mapping engine.

This module deliberately knows nothing about Hugging Face, PyTorch, JAX, Flax,
or any particular model. It maps one flat parameter dictionary into another
using explicit rules, validates keys and shapes, and can infer a last-two-dim
transpose where the declared mapping makes that unambiguous.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol, assert_never

from returns.result import Failure, Result, Success

type Shape = tuple[int, ...]


class TensorLike(Protocol):
    """Small structural protocol for tensors/arrays used by the mapper."""

    ndim: int

    @property
    def shape(self) -> Sequence[int]: ...


class TransformKind(Enum):
    """Supported layout transforms.

    AUTO is resolved before application. It infers identity or a transpose of
    the last two dimensions only when the tensor shapes make the choice unique.
    """

    IDENTITY = "identity"
    TRANSPOSE_LAST_TWO = "transpose_last_two"
    AUTO = "auto"


# Small DSL aliases. These keep model-specific rule files readable while
# preserving TransformKind as the canonical internal representation.
IDENTITY = TransformKind.IDENTITY
TRANSPOSE_LAST_TWO = TransformKind.TRANSPOSE_LAST_TWO
AUTO = TransformKind.AUTO


@dataclass(frozen=True, slots=True)
class MappingRule[DstKey: Hashable, SrcKey: Hashable]:
    """A single declared mapping from source parameter to destination parameter."""

    dst: DstKey
    src: SrcKey
    transform: TransformKind = TransformKind.AUTO


def rule[DstKey: Hashable, SrcKey: Hashable](
    dst: DstKey,
    src: SrcKey,
    transform: TransformKind = AUTO,
) -> MappingRule[DstKey, SrcKey]:
    """Small declarative helper for model-specific mapping tables.

    Equivalent to ``MappingRule(dst=dst, src=src, transform=transform)``, but
    reads better in architecture-specific rule files.
    """

    return MappingRule(dst=dst, src=src, transform=transform)


@dataclass(frozen=True, slots=True)
class MappingError[DstKey, SrcKey]:
    kind: Literal["missing_src", "missing_dst", "shape_mismatch", "ambiguous_transform"]
    dst: DstKey
    src: SrcKey | None = None
    dst_shape: Shape | None = None
    src_shape: Shape | None = None
    requested_transform: TransformKind | None = None


@dataclass(frozen=True, slots=True)
class ResolvedRule[DstKey, SrcKey, DstValue, SrcValue]:
    """A mapping rule whose keys exist and whose transform is concrete."""

    dst: DstKey
    src: SrcKey
    dst_value: DstValue
    src_value: SrcValue
    transform: TransformKind


def shape_of(x: TensorLike) -> Shape:
    return tuple(x.shape)


def transpose_last_two_shape(shape: Shape) -> Shape:
    if len(shape) < 2:
        return shape
    return shape[:-2] + (shape[-1], shape[-2])


def infer_transform(
    src_shape: Shape,
    dst_shape: Shape,
    requested: TransformKind,
) -> TransformKind | None:
    """Resolve a transform from shapes, returning None for mismatches or ambiguity."""

    match requested:
        case TransformKind.IDENTITY:
            return TransformKind.IDENTITY if src_shape == dst_shape else None

        case TransformKind.TRANSPOSE_LAST_TWO:
            if len(src_shape) >= 2 and transpose_last_two_shape(src_shape) == dst_shape:
                return TransformKind.TRANSPOSE_LAST_TWO
            return None

        case TransformKind.AUTO:
            identity = src_shape == dst_shape
            transpose = (
                len(src_shape) >= 2 and transpose_last_two_shape(src_shape) == dst_shape
            )
            if identity == transpose:
                return None
            return TransformKind.IDENTITY if identity else TransformKind.TRANSPOSE_LAST_TWO

        case _:
            assert_never(requested)


def resolve_rule[
    DstKey: Hashable,
    SrcKey: Hashable,
    DstValue: TensorLike,
    SrcValue: TensorLike,
](
    src: Mapping[SrcKey, SrcValue],
    dst: Mapping[DstKey, DstValue],
    rule: MappingRule[DstKey, SrcKey],
) -> Result[
    ResolvedRule[DstKey, SrcKey, DstValue, SrcValue],
    MappingError[DstKey, SrcKey],
]:
    if rule.src not in src:
        return Failure(
            MappingError(
                kind="missing_src",
                dst=rule.dst,
                src=rule.src,
                requested_transform=rule.transform,
            )
        )

    if rule.dst not in dst:
        return Failure(
            MappingError(
                kind="missing_dst",
                dst=rule.dst,
                src=rule.src,
                requested_transform=rule.transform,
            )
        )

    src_value = src[rule.src]
    dst_value = dst[rule.dst]
    src_shape = shape_of(src_value)
    dst_shape = shape_of(dst_value)

    transform = infer_transform(src_shape, dst_shape, rule.transform)
    if transform is None:
        # Both identity and last-two transpose are shape-compatible here.
        ambiguous = (
            rule.transform is TransformKind.AUTO
            and len(src_shape) >= 2
            and src_shape == dst_shape
            and src_shape[-2] == src_shape[-1]
        )
        return Failure(
            MappingError(
                kind="ambiguous_transform" if ambiguous else "shape_mismatch",
                dst=rule.dst,
                src=rule.src,
                dst_shape=dst_shape,
                src_shape=src_shape,
                requested_transform=rule.transform,
            )
        )

    return Success(
        ResolvedRule(
            dst=rule.dst,
            src=rule.src,
            dst_value=dst_value,
            src_value=src_value,
            transform=transform,
        )
    )


def partition_results[T, E](xs: Sequence[Result[T, E]]) -> tuple[list[T], list[E]]:
    oks: list[T] = []
    errs: list[E] = []

    for x in xs:
        match x:
            case Success(value):
                oks.append(value)
            case Failure(error):
                errs.append(error)

    return oks, errs


def resolve_rules[
    DstKey: Hashable,
    SrcKey: Hashable,
    DstValue: TensorLike,
    SrcValue: TensorLike,
](
    src: Mapping[SrcKey, SrcValue],
    dst: Mapping[DstKey, DstValue],
    rules: Sequence[MappingRule[DstKey, SrcKey]],
) -> Result[
    list[ResolvedRule[DstKey, SrcKey, DstValue, SrcValue]],
    list[MappingError[DstKey, SrcKey]],
]:
    resolved, errors = partition_results([resolve_rule(src, dst, rule) for rule in rules])
    if errors:
        return Failure(errors)
    return Success(resolved)


def apply_resolved_rules[
    DstKey: Hashable,
    SrcKey: Hashable,
    DstValue,
    SrcValue,
](
    dst: Mapping[DstKey, DstValue],
    resolved: Sequence[ResolvedRule[DstKey, SrcKey, DstValue, SrcValue]],
    *,
    convert: Callable[[SrcValue], DstValue],
    transpose_last_two: Callable[[SrcValue], SrcValue],
) -> dict[DstKey, DstValue]:
    """Apply already-resolved rules, preserving unmapped destination values."""

    out = dict(dst)

    for rule in resolved:
        src_value = rule.src_value

        match rule.transform:
            case TransformKind.IDENTITY:
                value = src_value
            case TransformKind.TRANSPOSE_LAST_TWO:
                value = transpose_last_two(src_value)
            case TransformKind.AUTO:
                raise RuntimeError("AUTO must be resolved before applying rules")
            case _:
                assert_never(rule.transform)

        out[rule.dst] = convert(value)

    return out


def map_param_dict[
    DstKey: Hashable,
    SrcKey: Hashable,
    DstValue: TensorLike,
    SrcValue: TensorLike,
](
    *,
    src: Mapping[SrcKey, SrcValue],
    dst: Mapping[DstKey, DstValue],
    rules: Sequence[MappingRule[DstKey, SrcKey]],
    convert: Callable[[SrcValue], DstValue],
    transpose_last_two: Callable[[SrcValue], SrcValue],
) -> Result[dict[DstKey, DstValue], list[MappingError[DstKey, SrcKey]]]:
    """Resolve and apply mapping rules in one Result-returning operation."""

    return resolve_rules(src, dst, rules).map(
        lambda resolved: apply_resolved_rules(
            dst,
            resolved,
            convert=convert,
            transpose_last_two=transpose_last_two,
        )
    )


def reverse_rules[
    DstKey: Hashable,
    SrcKey: Hashable,
](rules: Sequence[MappingRule[DstKey, SrcKey]]) -> list[MappingRule[SrcKey, DstKey]]:
    """Reverse mapping direction.

    Identity and last-two transpose are both their own inverse, so the same
    transform request is valid in the reverse direction. AUTO is re-inferred.
    """

    return [MappingRule(dst=rule.src, src=rule.dst, transform=rule.transform) for rule in rules]


def format_mapping_error[DstKey, SrcKey](error: MappingError[DstKey, SrcKey]) -> str:
    match error.kind:
        case "missing_src":
            return f"missing source for destination {error.dst!r}: source key {error.src!r}"
        case "missing_dst":
            return (
                f"missing destination for destination key {error.dst!r}: source key {error.src!r}"
            )
        case "shape_mismatch":
            return (
                "shape mismatch for mapping: "
                f"dst={error.dst!r} dst_shape={error.dst_shape}, "
                f"src={error.src!r} src_shape={error.src_shape}, "
                f"requested_transform={error.requested_transform}"
            )
        case "ambiguous_transform":
            return (
                "ambiguous AUTO transform for mapping: "
                f"dst={error.dst!r} dst_shape={error.dst_shape}, "
                f"src={error.src!r} src_shape={error.src_shape}; "
                "specify IDENTITY or TRANSPOSE_LAST_TWO explicitly"
            )
        case _:
            assert_never(error.kind)


def format_mapping_errors[DstKey, SrcKey](errors: Sequence[MappingError[DstKey, SrcKey]]) -> str:
    assert errors, "format_mapping_errors called with empty errors"
    return "\n".join(format_mapping_error(error) for error in errors)
