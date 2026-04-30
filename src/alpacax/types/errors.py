from dataclasses import dataclass
from typing import override

from pydantic import ValidationError
from returns.result import Result, ResultE


@dataclass(frozen=True, slots=True)
class RowError(Exception):
    row_index: int
    error: Exception

    @override
    def __str__(self) -> str:
        return f"row {self.row_index}: {format_error(self.error)}"


@dataclass(frozen=True, slots=True)
class PipelineError(Exception):
    error: Exception

    @override
    def __str__(self) -> str:
        return format_error(self.error)


type DatasetError = RowError | PipelineError


def format_error(error: Exception) -> str:
    if isinstance(error, ValidationError):
        try:
            errors = error.errors()
            if errors:
                msg = errors[0].get("msg")
                return msg or str(error)  # fallback if "msg" key is absent or empty
        except Exception:
            pass

    return str(error)


def annotate_failure[T](
    result: ResultE[T],
    row_index: int,
) -> Result[T, RowError]:
    return result.alt(
        lambda error: error if isinstance(error, RowError) else RowError(row_index, error)
    )


def to_dataset_error(error: Exception) -> DatasetError:
    if isinstance(error, (RowError, PipelineError)):
        return error
    return PipelineError(error)
