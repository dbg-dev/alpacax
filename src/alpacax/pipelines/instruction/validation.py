from typing import Any, cast

from returns.result import safe

from alpacax.types.models import Instruction


@safe
def validate_raw_dataset(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("expected dataset JSON to be a list")

    rows = cast(list[Any], raw)
    invalid_rows = [index for index, row in enumerate(rows) if not isinstance(row, dict)]

    if invalid_rows:
        preview = invalid_rows[:10]
        raise ValueError(
            f"expected every dataset row to be an object/dict; invalid row indices: {preview}"
        )

    return cast(list[dict[str, Any]], raw)


@safe
def parse_instruction(raw: dict[str, Any]) -> Instruction:
    return Instruction.model_validate(raw)
