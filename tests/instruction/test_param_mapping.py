from __future__ import annotations

import numpy as np
import pytest
from returns.result import Failure, Success

from alpacax.params.mapping import (
    MappingError,
    MappingRule,
    TensorLike,
    TransformKind,
    format_mapping_error,
    format_mapping_errors,
    infer_transform,
    map_param_dict,
    resolve_rules,
    reverse_rules,
    shape_of,
    transpose_last_two_shape,
)


def test_tensorlike_accepts_numpy_and_shape_only_objects() -> None:
    class ShapeOnly:
        @property
        def shape(self) -> tuple[int, ...]:
            return (2, 3)

    array: TensorLike = np.zeros((2, 3))
    shape_only: TensorLike = ShapeOnly()

    assert shape_of(array) == (2, 3)
    assert shape_of(shape_only) == (2, 3)


def test_transpose_last_two_shape():
    assert transpose_last_two_shape((2, 3)) == (3, 2)
    assert transpose_last_two_shape((4, 2, 3)) == (4, 3, 2)
    assert transpose_last_two_shape((3,)) == (3,)


def test_infer_transform_identity():
    assert infer_transform((2, 3), (2, 3), TransformKind.AUTO) == TransformKind.IDENTITY


def test_infer_transform_auto_transpose_last_two():
    assert infer_transform((2, 3), (3, 2), TransformKind.AUTO) == TransformKind.TRANSPOSE_LAST_TWO


def test_infer_transform_auto_rejects_ambiguous_square_matrix():
    assert infer_transform((2, 2), (2, 2), TransformKind.AUTO) is None
    assert infer_transform((3, 2, 2), (3, 2, 2), TransformKind.AUTO) is None


def test_infer_transform_vector_identity_but_not_transpose():
    assert infer_transform((3,), (3,), TransformKind.AUTO) is TransformKind.IDENTITY
    assert infer_transform((3,), (3,), TransformKind.TRANSPOSE_LAST_TWO) is None


def test_resolve_rules_reports_missing_source():
    src = {"a": np.zeros((2, 3))}
    dst = {"x": np.zeros((2, 3))}
    rules = [MappingRule(dst="x", src="missing")]

    result = resolve_rules(src, dst, rules)

    assert isinstance(result, Failure)
    assert result.failure() == [
        MappingError(
            kind="missing_src", dst="x", src="missing", requested_transform=TransformKind.AUTO
        )
    ]


def test_resolve_rules_reports_missing_destination():
    src = {"a": np.zeros((2, 3))}
    dst = {"x": np.zeros((2, 3))}
    rules = [MappingRule(dst="missing", src="a")]

    result = resolve_rules(src, dst, rules)

    assert isinstance(result, Failure)
    assert result.failure() == [
        MappingError(
            kind="missing_dst", dst="missing", src="a", requested_transform=TransformKind.AUTO
        )
    ]


def test_resolve_rules_reports_shape_mismatch():
    src = {"a": np.zeros((2, 3))}
    dst = {"x": np.zeros((4, 5))}
    rules = [MappingRule(dst="x", src="a")]

    result = resolve_rules(src, dst, rules)

    assert isinstance(result, Failure)
    [error] = result.failure()
    assert error.kind == "shape_mismatch"
    assert error.src_shape == (2, 3)
    assert error.dst_shape == (4, 5)


@pytest.mark.parametrize("shape", [(2, 2), (3, 2, 2)])
def test_auto_square_matrix_returns_ambiguity_error(shape):
    src = {"a": np.arange(np.prod(shape)).reshape(shape)}
    dst = {"x": np.zeros(shape)}

    result = resolve_rules(src, dst, [MappingRule(dst="x", src="a")])

    assert isinstance(result, Failure)
    [error] = result.failure()
    assert error == MappingError(
        kind="ambiguous_transform",
        dst="x",
        src="a",
        dst_shape=shape,
        src_shape=shape,
        requested_transform=TransformKind.AUTO,
    )
    assert "specify IDENTITY or TRANSPOSE_LAST_TWO" in format_mapping_error(error)


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        (TransformKind.IDENTITY, np.array([[1, 2], [3, 4]])),
        (TransformKind.TRANSPOSE_LAST_TWO, np.array([[1, 3], [2, 4]])),
    ],
)
def test_explicit_transform_maps_square_matrix(transform, expected):
    src = {"a": np.array([[1, 2], [3, 4]])}
    dst = {"x": np.zeros((2, 2))}

    result = map_param_dict(
        src=src,
        dst=dst,
        rules=[MappingRule(dst="x", src="a", transform=transform)],
        convert=lambda x: x,
        transpose_last_two=lambda x: x.T,
    )

    assert isinstance(result, Success)
    np.testing.assert_array_equal(result.unwrap()["x"], expected)


def test_explicit_transpose_rejects_vector():
    src = {"a": np.array([1, 2, 3])}
    dst = {"x": np.zeros((3,))}

    result = resolve_rules(
        src, dst, [MappingRule(dst="x", src="a", transform=TransformKind.TRANSPOSE_LAST_TWO)]
    )

    assert isinstance(result, Failure)
    [error] = result.failure()
    assert error.kind == "shape_mismatch"


def test_map_param_dict_preserves_unmapped_destination_values():
    src = {"a": np.array([1, 2, 3])}
    dst = {"x": np.array([0, 0, 0]), "y": np.array([9])}
    rules = [MappingRule(dst="x", src="a")]

    result = map_param_dict(
        src=src,
        dst=dst,
        rules=rules,
        convert=lambda x: x,
        transpose_last_two=lambda x: x.T,
    )

    assert isinstance(result, Success)
    mapped = result.unwrap()
    np.testing.assert_array_equal(mapped["x"], np.array([1, 2, 3]))
    np.testing.assert_array_equal(mapped["y"], np.array([9]))


def test_map_param_dict_applies_auto_transpose():
    src = {"a": np.array([[1, 2, 3], [4, 5, 6]])}
    dst = {"x": np.zeros((3, 2))}
    rules = [MappingRule(dst="x", src="a")]

    result = map_param_dict(
        src=src,
        dst=dst,
        rules=rules,
        convert=lambda x: x,
        transpose_last_two=lambda x: x.T,
    )

    assert isinstance(result, Success)
    np.testing.assert_array_equal(result.unwrap()["x"], src["a"].T)


def test_explicit_identity_does_not_auto_transpose():
    src = {"a": np.zeros((2, 3))}
    dst = {"x": np.zeros((3, 2))}
    rules = [MappingRule(dst="x", src="a", transform=TransformKind.IDENTITY)]

    result = map_param_dict(
        src=src,
        dst=dst,
        rules=rules,
        convert=lambda x: x,
        transpose_last_two=lambda x: x.T,
    )

    assert isinstance(result, Failure)
    [error] = result.failure()
    assert error.kind == "shape_mismatch"
    assert error.requested_transform == TransformKind.IDENTITY


def test_reverse_rules_swaps_source_and_destination():
    rules = [MappingRule(dst="local", src="hf", transform=TransformKind.AUTO)]

    assert reverse_rules(rules) == [
        MappingRule(dst="hf", src="local", transform=TransformKind.AUTO)
    ]


def test_strict_mode_reports_all_missing_coverage_in_one_failure() -> None:
    src = {"a": np.array([1, 2]), "extra": np.array([3])}
    dst = {"x": np.array([0, 0]), "untouched": np.array([9])}
    rules = [MappingRule(dst="x", src="a")]

    result = map_param_dict(
        src=src,
        dst=dst,
        rules=rules,
        convert=lambda x: x,
        transpose_last_two=lambda x: x.T,
        strict=True,
    )

    assert isinstance(result, Failure)
    assert result.failure() == [
        MappingError(kind="unmapped_dst", dst="untouched"),
        MappingError(kind="unused_src", dst=None, src="extra"),
    ]
    message = format_mapping_errors(result.failure())
    assert "unmapped destination parameter: 'untouched'" in message
    assert "unused source parameter: 'extra'" in message


def test_strict_mode_honours_explicit_exceptions_for_tied_weights() -> None:
    src = {"a": np.array([1]), "lm_head.weight": np.array([1])}
    dst = {"x": np.array([0]), "optional": np.array([9])}
    result = map_param_dict(
        src=src,
        dst=dst,
        rules=[MappingRule(dst="x", src="a")],
        convert=lambda x: x,
        transpose_last_two=lambda x: x.T,
        strict=True,
        allow_unmapped_dst=("optional",),
        allow_unused_src=("lm_head.weight",),
    )

    assert isinstance(result, Success)
    np.testing.assert_array_equal(result.unwrap()["x"], np.array([1]))
    np.testing.assert_array_equal(result.unwrap()["optional"], np.array([9]))


def test_strict_mode_accepts_shared_source_keys() -> None:
    src = {"shared": np.array([3])}
    dst = {"one": np.array([0]), "two": np.array([0])}
    result = resolve_rules(
        src,
        dst,
        [
            MappingRule(dst="one", src="shared"),
            MappingRule(dst="two", src="shared"),
        ],
        strict=True,
    )
    assert isinstance(result, Success)
    assert len(result.unwrap()) == 2


def test_strict_mode_combines_rule_errors_with_coverage_errors() -> None:
    src = {"unused": np.array([1])}
    dst = {"mapped": np.array([0]), "unmapped": np.array([0])}
    result = resolve_rules(
        src,
        dst,
        [MappingRule(dst="mapped", src="missing")],
        strict=True,
    )
    assert isinstance(result, Failure)
    assert [error.kind for error in result.failure()] == [
        "missing_src",
        "unmapped_dst",
        "unused_src",
    ]


def test_strict_failure_does_not_apply_any_updates() -> None:
    src = {"a": np.array([1]), "extra": np.array([2])}
    dst = {"x": np.array([0])}
    conversions: list[bool] = []

    def convert(value: np.ndarray) -> np.ndarray:
        conversions.append(True)
        return value

    result = map_param_dict(
        src=src,
        dst=dst,
        rules=[MappingRule(dst="x", src="a")],
        convert=convert,
        transpose_last_two=lambda x: x.T,
        strict=True,
    )
    assert isinstance(result, Failure)
    assert not conversions
    np.testing.assert_array_equal(dst["x"], np.array([0]))


def test_permissive_mode_still_accepts_extra_source_and_destination_keys() -> None:
    result = map_param_dict(
        src={"a": np.array([1]), "extra": np.array([99])},
        dst={"x": np.array([0]), "random_init": np.array([42])},
        rules=[MappingRule(dst="x", src="a")],
        convert=lambda x: x,
        transpose_last_two=lambda x: x.T,
    )
    assert isinstance(result, Success)
    np.testing.assert_array_equal(result.unwrap()["random_init"], np.array([42]))
