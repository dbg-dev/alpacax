from unittest.mock import Mock, call

import pytest

from alpacax.pipelines.instruction.transforms import tokenize_sample
from alpacax.types.models import AlpacaSample, TokenizedSample

## Happy


def test_tokenize_sample_basic():
    tokenizer = Mock()
    tokenizer.encode.side_effect = [
        [1, 2],  # prompt
        [3, 4],  # response
    ]

    sample = AlpacaSample(prompt="p", response="r")

    result = tokenize_sample(
        sample,
        tokenizer,
        eos_token_id=None,
        add_eos_token=False,
    )

    assert result == TokenizedSample(
        prompt_tokens=[1, 2],
        response_tokens=[3, 4],
    )

    tokenizer.encode.assert_any_call("p")
    tokenizer.encode.assert_any_call("r")
    assert tokenizer.encode.call_count == 2


## EOS checks


def test_tokenize_sample_adds_eos_token():
    tokenizer = Mock()
    tokenizer.encode.side_effect = [
        [1],  # prompt
        [2, 3],  # response
    ]

    sample = AlpacaSample(prompt="p", response="r")

    result = tokenize_sample(
        sample,
        tokenizer,
        eos_token_id=99,
        add_eos_token=True,
    )

    assert result.response_tokens == [2, 3, 99]


def test_tokenize_sample_does_not_add_eos_when_flag_false():
    tokenizer = Mock()
    tokenizer.encode.side_effect = [
        [1],
        [2],
    ]

    result = tokenize_sample(
        AlpacaSample(prompt="p", response="r"),
        tokenizer,
        eos_token_id=99,
        add_eos_token=False,
    )

    assert result.response_tokens == [2]


def test_tokenize_sample_does_not_add_eos_when_id_none():
    tokenizer = Mock()
    tokenizer.encode.side_effect = [
        [1],
        [2],
    ]

    result = tokenize_sample(
        AlpacaSample(prompt="p", response="r"),
        tokenizer,
        eos_token_id=None,
        add_eos_token=True,
    )

    assert result.response_tokens == [2]


## Empty response


def test_tokenize_sample_raises_if_response_empty():
    tokenizer = Mock()
    tokenizer.encode.side_effect = [
        [1, 2],  # prompt
        [],  # response
    ]

    with pytest.raises(ValueError, match="response produced no tokens"):
        tokenize_sample(
            AlpacaSample(prompt="p", response="r"),
            tokenizer,
            eos_token_id=None,
            add_eos_token=False,
        )


## empty failure


def test_tokenize_sample_empty_response_becomes_valid_with_eos():
    tokenizer = Mock()
    tokenizer.encode.side_effect = [
        [1],
        [],  # response initially empty
    ]

    result = tokenize_sample(
        AlpacaSample(prompt="p", response="r"),
        tokenizer,
        eos_token_id=99,
        add_eos_token=True,
    )

    assert result.response_tokens == [99]


## call order


def test_tokenize_sample_call_order():
    tokenizer = Mock()
    tokenizer.encode.side_effect = [[1], [2]]

    tokenize_sample(
        AlpacaSample(prompt="p", response="r"),
        tokenizer,
        eos_token_id=None,
        add_eos_token=False,
    )

    assert tokenizer.encode.call_args_list == [
        call("p"),
        call("r"),
    ]
