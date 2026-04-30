import pytest

from alpacax.pipelines.instruction.transforms import prepare_sample
from alpacax.types.models import PaddedSample, TokenizedSample


def test_prepare_sample_creates_shifted_input_labels_and_attention_mask():
    sample = TokenizedSample(
        prompt_tokens=[10, 11],
        response_tokens=[20, 21],
    )

    result = prepare_sample(
        sample,
        truncation="skip",
        min_seq_len=2,
        max_seq_len=6,
        pad_token=0,
        ignore_index=-100,
        mask_prompt_labels=False,
    )

    assert result == PaddedSample(
        input_ids=[10, 11, 20, 0, 0],
        labels=[11, 20, 21, -100, -100],
        attention_mask=[1, 1, 1, 0, 0],
    )


def test_prepare_sample_masks_prompt_labels_but_keeps_first_response_prediction():
    sample = TokenizedSample(
        prompt_tokens=[10, 11, 12],
        response_tokens=[20, 21],
    )

    result = prepare_sample(
        sample,
        truncation="skip",
        min_seq_len=2,
        max_seq_len=6,
        pad_token=0,
        ignore_index=-100,
        mask_prompt_labels=True,
    )

    assert result.input_ids == [10, 11, 12, 20, 0]
    assert result.labels == [-100, -100, 20, 21, -100]
    assert result.attention_mask == [1, 1, 1, 1, 0]


## truncation checks


def test_prepare_sample_truncates_when_configured():
    sample = TokenizedSample(
        prompt_tokens=[10, 11],
        response_tokens=[20, 21, 22, 23],
    )

    result = prepare_sample(
        sample,
        truncation="truncate",
        min_seq_len=2,
        max_seq_len=4,
        pad_token=0,
        ignore_index=-100,
        mask_prompt_labels=False,
    )

    # token_list becomes [10, 11, 20, 21]
    assert result.input_ids == [10, 11, 20]
    assert result.labels == [11, 20, 21]
    assert result.attention_mask == [1, 1, 1]


## Failure cases


def test_prepare_sample_raises_when_too_short():
    sample = TokenizedSample(
        prompt_tokens=[10],
        response_tokens=[],
    )

    with pytest.raises(ValueError, match="minimum is 2"):
        prepare_sample(
            sample,
            truncation="skip",
            min_seq_len=2,
            max_seq_len=6,
            pad_token=0,
            ignore_index=-100,
            mask_prompt_labels=True,
        )


def test_prepare_sample_raises_when_too_long_and_truncation_is_skip():
    sample = TokenizedSample(
        prompt_tokens=[10, 11],
        response_tokens=[20, 21, 22],
    )

    with pytest.raises(ValueError, match="maximum is 4"):
        prepare_sample(
            sample,
            truncation="skip",
            min_seq_len=2,
            max_seq_len=4,
            pad_token=0,
            ignore_index=-100,
            mask_prompt_labels=True,
        )


def test_prepare_sample_raises_when_truncation_removes_all_response_tokens():
    sample = TokenizedSample(
        prompt_tokens=[10, 11, 12, 13],
        response_tokens=[20, 21],
    )

    with pytest.raises(ValueError, match="truncation removed all response tokens"):
        prepare_sample(
            sample,
            truncation="truncate",
            min_seq_len=2,
            max_seq_len=4,
            pad_token=0,
            ignore_index=-100,
            mask_prompt_labels=True,
        )


## edge case


def test_prepare_sample_single_prompt_token_masking_keeps_first_label():
    sample = TokenizedSample(
        prompt_tokens=[10],
        response_tokens=[20, 21],
    )

    result = prepare_sample(
        sample,
        truncation="skip",
        min_seq_len=2,
        max_seq_len=5,
        pad_token=0,
        ignore_index=-100,
        mask_prompt_labels=True,
    )

    assert result.input_ids == [10, 20, 0, 0]
    assert result.labels == [20, 21, -100, -100]
    assert result.attention_mask == [1, 1, 0, 0]
