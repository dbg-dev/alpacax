from __future__ import annotations

import numpy as np
from returns.result import Failure, Success

from alpacax.params.mapping import (
    MappingError,
    MappingRule,
    TransformKind,
    infer_transform,
    map_param_dict,
    resolve_rules,
    reverse_rules,
    transpose_last_two_shape,
)


def test_transpose_last_two_shape():
    assert transpose_last_two_shape((2, 3)) == (3, 2)
    assert transpose_last_two_shape((4, 2, 3)) == (4, 3, 2)
    assert transpose_last_two_shape((3,)) == (3,)


def test_infer_transform_identity():
    assert infer_transform((2, 3), (2, 3), TransformKind.AUTO) == TransformKind.IDENTITY


def test_infer_transform_auto_transpose_last_two():
    assert infer_transform((2, 3), (3, 2), TransformKind.AUTO) == TransformKind.TRANSPOSE_LAST_TWO


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
