from functools import partial

from returns.pipeline import flow
from returns.pointfree import bind, map_
from returns.result import Failure, ResultE, safe

from alpacax.pipelines.instruction.config import PipelineConfig
from alpacax.pipelines.instruction.pipeio import fetch_raw
from alpacax.pipelines.instruction.stages import (
    drain,
    process_instruction,
    shuffle_split,
)
from alpacax.pipelines.instruction.validation import validate_raw_dataset
from alpacax.types.errors import DatasetError, annotate_failure
from alpacax.types.models import (
    DatasetSplit,
    PaddedSample,
)
from alpacax.types.protocols import Tokenizer


@safe
def _process_instructions(
    data: list[dict[str, str]],
    tokenizer: Tokenizer,
    config: PipelineConfig,
) -> tuple[list[PaddedSample], list[DatasetError]]:

    samples, errors = drain(
        annotate_failure(
            process_instruction(datum, tokenizer=tokenizer, config=config),
            row_index=index,
        )
        for index, datum in enumerate(data)
    )

    if not samples:
        error_text = "; ".join(str(error) for error in errors)
        raise ValueError(f"no valid samples after processing. Errors: {error_text}")

    return samples, errors


def process_instructions(
    tokenizer: Tokenizer,
    config: PipelineConfig,
):
    return partial(_process_instructions, tokenizer=tokenizer, config=config)


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
        map_(shuffle_split(cfg)),
    )
