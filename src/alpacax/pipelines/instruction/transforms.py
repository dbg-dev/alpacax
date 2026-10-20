import random
from typing import Literal

from alpacax.types.models import (
    AlpacaSample,
    DatasetSplit,
    Instruction,
    PaddedSample,
    PreparedSample,
    SplitConfig,
    TokenizedSample,
)
from alpacax.types.protocols import Tokenizer


def format_alpaca(instruction: Instruction) -> AlpacaSample:
    if instruction.input:
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction.instruction}\n\n"
            f"### Input:\n{instruction.input}\n\n"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction.instruction}\n\n"
        )
    response = f"### Response:\n{instruction.output}"
    return AlpacaSample(prompt=prompt, response=response)


def make_input_label[T](token_list: list[T]) -> tuple[list[T], list[T]]:
    """
    Shift token_list by one to create an input/label pair.

        token_list: [a, b, c, d]
        input:      [a, b, c]
        label:      [b, c, d]
    """
    return token_list[:-1], token_list[1:]


def pad_to(tokens: list[int], length: int, pad_value: int) -> list[int]:
    deficit = length - len(tokens)
    if deficit < 0:
        raise ValueError(f"token list length {len(tokens)} exceeds target {length}")
    return tokens + [pad_value] * deficit


def shuffle[T](
    samples: list[T],
    seed: int,
) -> list[T]:

    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    return shuffled


def split_samples[T, U](
    samples: list[T],
    skipped: list[U] | None,
    weights: SplitConfig,
) -> DatasetSplit[T, U]:
    n = len(samples)
    train_end = round(n * weights.train)
    validation_end = n - round(n * weights.test)  # derive from test, not val

    return DatasetSplit[T, U](
        train=samples[:train_end],
        validation=samples[train_end:validation_end],
        test=samples[validation_end:],
        skipped=skipped,
    )


def prepare_unpadded_sample(
    sample: TokenizedSample,
    *,
    truncation: Literal["skip", "truncate"],
    min_seq_len: int,
    max_seq_len: int,
    ignore_index: int,
    mask_prompt_labels: bool,
) -> PreparedSample:
    """Create shifted input/target tokens without padding.

    The label predicting the first response token remains trainable even when
    preceding prompt positions are ignored. Sequence lengths stay bounded by
    max_seq_len - 1 after shifting, matching the legacy padded preparation.
    """
    token_list = sample.token_list

    if len(token_list) < min_seq_len:
        raise ValueError(f"sample has {len(token_list)} tokens; minimum is {min_seq_len}")

    if len(token_list) > max_seq_len:
        if truncation == "skip":
            raise ValueError(
                f"sample has {len(token_list)} tokens; maximum is {max_seq_len}"
            )
        token_list = token_list[:max_seq_len]

        if len(token_list) <= len(sample.prompt_tokens):
            raise ValueError("truncation removed all response tokens")

    inputs, labels = make_input_label(token_list)

    if mask_prompt_labels:
        prompt_label_cutoff = max(len(sample.prompt_tokens) - 1, 0)
        labels = [
            ignore_index if idx < prompt_label_cutoff else label for idx, label in enumerate(labels)
        ]

    return PreparedSample(input_ids=inputs, labels=labels)


def prepare_sample(
    sample: TokenizedSample,
    *,
    truncation: Literal["skip", "truncate"],
    min_seq_len: int,
    max_seq_len: int,
    pad_token: int,
    ignore_index: int,
    mask_prompt_labels: bool,
) -> PaddedSample:
    """Legacy fixed-length output, derived from the unpadded preparation."""
    prepared = prepare_unpadded_sample(
        sample,
        truncation=truncation,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        ignore_index=ignore_index,
        mask_prompt_labels=mask_prompt_labels,
    )
    target_len = max_seq_len - 1
    attention_mask = [1] * len(prepared.input_ids)
    return PaddedSample(
        input_ids=pad_to(prepared.input_ids, target_len, pad_token),
        labels=pad_to(prepared.labels, target_len, ignore_index),
        attention_mask=pad_to(attention_mask, target_len, 0),
    )

def tokenize_sample(
    sample: AlpacaSample,
    tokenizer: Tokenizer,
    *,
    eos_token_id: int | None,
    add_eos_token: bool,
) -> TokenizedSample:
    prompt_tokens = tokenizer.encode(sample.prompt)
    response_tokens = tokenizer.encode(sample.response)

    if add_eos_token and eos_token_id is not None:
        response_tokens = response_tokens + [eos_token_id]

    if not response_tokens:
        raise ValueError("response produced no tokens")

    return TokenizedSample(
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
    )
