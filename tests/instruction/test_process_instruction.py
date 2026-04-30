from unittest.mock import Mock, patch

from returns.result import Failure, Success

from alpacax.pipelines.instruction.stages import process_instruction


def test_process_instruction_success():
    tokenizer = Mock()
    config = Mock()

    datum = {"instruction": "i", "input": "", "output": "o"}
    parsed = Mock(name="parsed")
    formatted = Mock(name="formatted")
    tokenized = Mock(name="tokenized")
    padded = Mock(name="padded")

    tokenize_fn = Mock(return_value=Success(tokenized))
    prepare_fn = Mock(return_value=Success(padded))

    with (
        patch(
            "alpacax.pipelines.instruction.stages.parse_instruction", return_value=Success(parsed)
        ) as parse_instruction,
        patch(
            "alpacax.pipelines.instruction.stages.format_alpaca", return_value=formatted
        ) as format_alpaca,
        patch(
            "alpacax.pipelines.instruction.stages.tokenize", return_value=tokenize_fn
        ) as tokenize,
        patch("alpacax.pipelines.instruction.stages.prepare", return_value=prepare_fn) as prepare,
    ):
        result = process_instruction(datum, tokenizer, config)

    assert result == Success(padded)

    parse_instruction.assert_called_once_with(datum)
    format_alpaca.assert_called_once_with(parsed)
    tokenize.assert_called_once_with(tokenizer, config)
    tokenize_fn.assert_called_once_with(formatted)
    prepare.assert_called_once_with(config)
    prepare_fn.assert_called_once_with(tokenized)


def test_process_instruction_short_circuits_on_parse_failure():
    tokenizer = Mock()
    config = Mock()
    error = ValueError("bad row")

    tokenize_fn = Mock()
    prepare_fn = Mock()

    with (
        patch(
            "alpacax.pipelines.instruction.stages.parse_instruction", return_value=Failure(error)
        ),
        patch("alpacax.pipelines.instruction.stages.format_alpaca") as format_alpaca,
        patch(
            "alpacax.pipelines.instruction.stages.tokenize", return_value=tokenize_fn
        ) as tokenize,
        patch("alpacax.pipelines.instruction.stages.prepare", return_value=prepare_fn) as prepare,
    ):
        result = process_instruction({"bad": "row"}, tokenizer, config)

    assert result == Failure(error)

    format_alpaca.assert_not_called()

    # Factories are still evaluated when building the flow.
    tokenize.assert_called_once_with(tokenizer, config)
    prepare.assert_called_once_with(config)

    # But their returned stage functions are not applied.
    tokenize_fn.assert_not_called()
    prepare_fn.assert_not_called()


def test_process_instruction_short_circuits_on_tokenize_failure():
    tokenizer = Mock()
    config = Mock()

    parsed = Mock(name="parsed")
    formatted = Mock(name="formatted")
    error = RuntimeError("tokenizer failed")

    tokenize_fn = Mock(return_value=Failure(error))
    prepare_fn = Mock()

    with (
        patch(
            "alpacax.pipelines.instruction.stages.parse_instruction", return_value=Success(parsed)
        ),
        patch("alpacax.pipelines.instruction.stages.format_alpaca", return_value=formatted),
        patch("alpacax.pipelines.instruction.stages.tokenize", return_value=tokenize_fn),
        patch("alpacax.pipelines.instruction.stages.prepare", return_value=prepare_fn) as prepare,
    ):
        result = process_instruction({"instruction": "i"}, tokenizer, config)

    assert result == Failure(error)

    tokenize_fn.assert_called_once_with(formatted)
    prepare.assert_called_once_with(config)
    prepare_fn.assert_not_called()


def test_process_instruction_short_circuits_on_prepare_failure():
    tokenizer = Mock()
    config = Mock()

    parsed = Mock(name="parsed")
    formatted = Mock(name="formatted")
    tokenized = Mock(name="tokenized")
    error = ValueError("too long")

    tokenize_fn = Mock(return_value=Success(tokenized))
    prepare_fn = Mock(return_value=Failure(error))

    with (
        patch(
            "alpacax.pipelines.instruction.stages.parse_instruction", return_value=Success(parsed)
        ),
        patch("alpacax.pipelines.instruction.stages.format_alpaca", return_value=formatted),
        patch("alpacax.pipelines.instruction.stages.tokenize", return_value=tokenize_fn),
        patch("alpacax.pipelines.instruction.stages.prepare", return_value=prepare_fn),
    ):
        result = process_instruction({"instruction": "i"}, tokenizer, config)

    assert result == Failure(error)

    tokenize_fn.assert_called_once_with(formatted)
    prepare_fn.assert_called_once_with(tokenized)
