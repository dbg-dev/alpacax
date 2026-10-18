from collections.abc import Callable, Iterable
from functools import partial

from returns.pipeline import flow
from returns.pointfree import bind, map_
from returns.result import Failure, Result, ResultE, Success, safe

from alpacax.pipelines.instruction.config import PipelineConfig
from alpacax.pipelines.instruction.transforms import (
    format_alpaca,
    prepare_sample,
    shuffle,
    split_samples,
    tokenize_sample,
)
from alpacax.pipelines.instruction.validation import parse_instruction
from alpacax.types.errors import DatasetError, to_dataset_error
from alpacax.types.models import DatasetSplit, PaddedSample, SplitConfig
from alpacax.types.protocols import Tokenizer

# ── Generator pipeline ────────────────────────────────────────────────────────


def tokenize(
    tokenizer: Tokenizer,
    config: PipelineConfig,
):
    return safe(
        partial(
            tokenize_sample,
            tokenizer=tokenizer,
            eos_token_id=config.eos_token_id,
            add_eos_token=config.add_eos_token,
        )
    )


def prepare(
    config: PipelineConfig,
):
    return safe(
        partial(
            prepare_sample,
            truncation=config.truncation,
            min_seq_len=config.min_seq_len,
            max_seq_len=config.max_seq_len,
            pad_token=config.pad_token,
            ignore_index=config.ignore_index,
            mask_prompt_labels=config.mask_prompt_labels,
        )
    )


def _shuffle_split(
    processed: tuple[list[PaddedSample], list[DatasetError]],
    seed: int,
    weights: SplitConfig,
) -> DatasetSplit[PaddedSample, DatasetError]:
    samples, errors = processed
    shuffled = shuffle(samples, seed)
    split = split_samples(shuffled, errors, weights)
    return split


def shuffle_split(
    config: PipelineConfig,
) -> Callable[
    [tuple[list[PaddedSample], list[DatasetError]]],
    DatasetSplit[PaddedSample, DatasetError],
]:
    return partial(_shuffle_split, seed=config.seed, weights=config.weights)


def process_instruction(
    datum: dict[str, str],
    tokenizer: Tokenizer,
    config: PipelineConfig,
) -> ResultE[PaddedSample]:
    return flow(
        datum,
        parse_instruction,
        map_(format_alpaca),  # pyright: ignore[reportUnknownArgumentType]
        bind(tokenize(tokenizer, config)),
        bind(prepare(config)),
    )


# ── Drain ─────────────────────────────────────────────────────────────────────


def drain[T](
    stream: Iterable[Result[T, Exception]],
) -> tuple[list[T], list[DatasetError]]:
    oks: list[T] = []
    errs: list[DatasetError] = []

    for result in stream:
        match result:
            case Success(value):
                oks.append(value)
            case Failure(error):
                errs.append(to_dataset_error(error))
            case _ as unreachable:
                raise AssertionError(f"Unreachable: {type(unreachable)}")

    return oks, errs
