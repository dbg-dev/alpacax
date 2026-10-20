from typing import Any, override
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails, InitErrorDetails, PydanticCustomError
from returns.result import Failure, Success

from alpacax.types.errors import RowError, annotate_failure, format_error

## annotate_failure


def test_annotate_failure_preserves_success_value():
    result = annotate_failure(Success("ok"), row_index=3)

    assert result == Success("ok")


def test_annotate_failure_wraps_failure_in_row_error():
    original = ValueError("bad row")

    result = annotate_failure(Failure(original), row_index=7)

    assert isinstance(result.failure(), RowError)
    assert result.failure().row_index == 7
    assert result.failure().error is original


def test_annotate_failure_row_error_string_includes_row_and_message():
    result = annotate_failure(Failure(ValueError("missing field")), row_index=2)

    assert str(result.failure()) == "row 2: missing field"


def test_annotate_failure_preserves_original_validation_error():
    class ExampleModel(BaseModel):
        name: str

    with pytest.raises(ValidationError) as exc_info:
        ExampleModel()  # pyright: ignore[reportCallIssue]

    result = annotate_failure(Failure(exc_info.value), row_index=4)

    row_error = result.failure()
    assert isinstance(row_error, RowError)
    assert row_error.row_index == 4
    assert row_error.error is exc_info.value
    assert "row 4:" in str(row_error)
    assert "Field required" in str(row_error)


def test_annotate_failure_does_not_double_wrap_row_error():
    original = RowError(row_index=1, error=ValueError("bad"))

    result = annotate_failure(Failure(original), row_index=9)

    row_error = result.failure()
    assert row_error is original  # returned as-is
    assert row_error.row_index == 1  # original index preserved, not 9


## format errors


def test_format_error_returns_regular_exception_message():
    error = ValueError("bad row")

    assert format_error(error) == "bad row"


def test_format_error_returns_first_pydantic_validation_message():
    class ExampleModel(BaseModel):
        name: str

    with pytest.raises(ValidationError) as exc_info:
        ExampleModel()  # pyright: ignore[reportCallIssue]

    assert format_error(exc_info.value) == "Field required"


def test_format_error_falls_back_to_str_for_validation_error_without_errors():
    class EmptyValidationError(ValidationError):
        @override
        def errors(self, *args: Any, **kwargs: Any) -> list[ErrorDetails]:
            del args, kwargs
            return []

    error = EmptyValidationError.from_exception_data(
        title="EmptyValidationError",
        line_errors=[],
    )

    assert format_error(error) == str(error)


def test_format_error_falls_back_to_str_when_msg_key_absent():
    error = ValidationError.from_exception_data(
        title="Test",
        line_errors=[
            InitErrorDetails(
                type=PydanticCustomError("custom_type", "something went wrong"),
                loc=(),
                input=None,
            )
        ],
    )

    # Patch errors() to return a dict without a "msg" key
    with patch.object(type(error), "errors", return_value=[{"type": "custom_type"}]):
        result = format_error(error)

    assert result == str(error)
