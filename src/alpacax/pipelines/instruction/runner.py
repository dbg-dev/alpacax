from collections.abc import Callable
from functools import partial

from returns.pipeline import flow
from returns.pointfree import bind, map_
from returns.result import Failure, ResultE, safe

from alpacax.pipelines.instruction.config import PipelineConfig
from alpacax.pipelines.instruction.pipeio import fetch_raw
from alpacax.pipelines.instruction.stages import (
    drain,
    process_instruction,
    process_unpadded_instruction,
    shuffle_split,
    shuffle_split_unpadded,
)
from alpacax.pipelines.instruction.validation import validate_raw_dataset
from alpacax.types.errors import DatasetError, annotate_failure
from alpacax.types.models import (
    DatasetSplit,
    PaddedSample,
    PreparedSample,
)
from alpacax.types.protocols import Tokenizer


def _process_records[T](
    data: list[dict[str, str]],
    tokenizer: Tokenizer,
    config: PipelineConfig,
    process: Callable[[dict[str, str], Tokenizer, PipelineConfig], ResultE[T]],
) -> tuple[list[T], list[DatasetError]]:
    samples, errors = drain(
        annotate_failure(
            process(datum, tokenizer, config),
            row_index=index,
        )
        for index, datum in enumerate(data)
    )

    if not samples:
        error_text = "; ".join(str(error) for error in errors)
        raise ValueError(f"no valid samples after processing. Errors: {error_text}")

    return samples, errors


@safe
def _process_instructions(
    data: list[dict[str, str]],
    tokenizer: Tokenizer,
    config: PipelineConfig,
) -> tuple[list[PaddedSample], list[DatasetError]]:

    return _process_records(data, tokenizer, config, process_instruction)


def process_instructions(
    tokenizer: Tokenizer,
    config: PipelineConfig,
):
    return partial(_process_instructions, tokenizer=tokenizer, config=config)


@safe
def _process_unpadded_instructions(
    data: list[dict[str, str]],
    tokenizer: Tokenizer,
    config: PipelineConfig,
) -> tuple[list[PreparedSample], list[DatasetError]]:
    return _process_records(data, tokenizer, config, process_unpadded_instruction)


def process_unpadded_instructions(
    tokenizer: Tokenizer,
    config: PipelineConfig,
):
    return partial(_process_unpadded_instructions, tokenizer=tokenizer, config=config)


def build_dataset(
    url: str,
    tokenizer: Tokenizer,
    config: PipelineConfig | None = None,
) -> ResultE[DatasetSplit[PaddedSample, DatasetError]]:

    try:
        cfg = config or PipelineConfig()
    except Exception as exc:
        return Failure(exc)

    _fetch = partial(fetch_raw, timeout_seconds=cfg.request_timeout_seconds)

    return flow(
        url,
        _fetch,
        bind(validate_raw_dataset),
        bind(process_instructions(tokenizer, cfg)),
        map_(shuffle_split(cfg)),  # pyright: ignore[reportUnknownArgumentType]
    )


def build_unpadded_dataset(
    url: str,
    tokenizer: Tokenizer,
    config: PipelineConfig | None = None,
) -> ResultE[DatasetSplit[PreparedSample, DatasetError]]:
    """Build compact, shifted instruction records for batch-time padding.

    The existing build_dataset() intentionally retains its PaddedSample result
    for compatibility. Neither function imports any ML training dependencies.
    """
    try:
        cfg = config or PipelineConfig()
    except Exception as exc:
        return Failure(exc)

    _fetch = partial(fetch_raw, timeout_seconds=cfg.request_timeout_seconds)

    return flow(
        url,
        _fetch,
        bind(validate_raw_dataset),
        bind(process_unpadded_instructions(tokenizer, cfg)),
        map_(shuffle_split_unpadded(cfg)),  # pyright: ignore[reportUnknownArgumentType]
    )
