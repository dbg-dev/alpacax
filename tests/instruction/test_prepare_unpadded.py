"""Unpadded instruction samples preserve the legacy shifted-label semantics."""

import pytest

from alpacax.pipelines.instruction.transforms import (
    prepare_sample,
    prepare_unpadded_sample,
)
from alpacax.types.models import PreparedSample, TokenizedSample


def test_unpadded_preserves_shift_and_first_response_target() -> None:
    sample = TokenizedSample(
        prompt_tokens=[10, 11, 12], response_tokens=[20, 21]
    )
    result = prepare_unpadded_sample(
        sample,
        truncation="skip",
        min_seq_len=2,
        max_seq_len=16,
        ignore_index=-7,
        mask_prompt_labels=True,
    )
    assert result == PreparedSample(
        input_ids=[10, 11, 12, 20],
        labels=[-7, -7, 20, 21],
    )
    assert len(result.input_ids) == 4  # No padding to max_seq_len - 1.


def test_unpadded_and_padded_have_identical_real_tokens() -> None:
    sample = TokenizedSample(prompt_tokens=[5], response_tokens=[6, 7, 8])
    for mask_prompt in (True, False):
        compact = prepare_unpadded_sample(
            sample,
            truncation="skip",
            min_seq_len=2,
            max_seq_len=12,
            ignore_index=-9,
            mask_prompt_labels=mask_prompt,
        )
        padded = prepare_sample(
            sample,
            truncation="skip",
            min_seq_len=2,
            max_seq_len=12,
            pad_token=0,
            ignore_index=-9,
            mask_prompt_labels=mask_prompt,
        )
        length = len(compact.input_ids)
        assert padded.input_ids[:length] == compact.input_ids
        assert padded.labels[:length] == compact.labels
        assert padded.attention_mask == [1] * length + [0] * (11 - length)
        assert padded.labels[length:] == [-9] * (11 - length)


def test_unpadded_truncation_preserves_response_target() -> None:
    compact = prepare_unpadded_sample(
        TokenizedSample(prompt_tokens=[10, 11], response_tokens=[20, 21, 22, 23]),
        truncation="truncate",
        min_seq_len=2,
        max_seq_len=4,
        ignore_index=-100,
        mask_prompt_labels=True,
    )
    assert compact.input_ids == [10, 11, 20]
    assert compact.labels == [-100, 20, 21]


def test_unpadded_rejects_prompt_only_truncation() -> None:
    with pytest.raises(ValueError, match="truncation removed all response tokens"):
        prepare_unpadded_sample(
            TokenizedSample(prompt_tokens=[10, 11, 12, 13], response_tokens=[20]),
            truncation="truncate",
            min_seq_len=2,
            max_seq_len=4,
            ignore_index=-100,
            mask_prompt_labels=True,
        )


def test_unpadded_rejects_samples_exceeding_limit_with_skip() -> None:
    with pytest.raises(ValueError, match="maximum is 3"):
        prepare_unpadded_sample(
            TokenizedSample(prompt_tokens=[10, 11], response_tokens=[20, 21]),
            truncation="skip",
            min_seq_len=2,
            max_seq_len=3,
            ignore_index=-100,
            mask_prompt_labels=True,
        )
