from unittest.mock import Mock, patch

from returns.result import Failure, Success

from alpacax.pipelines.instruction.runner import build_dataset


def test_build_dataset_success():
    tokenizer = Mock()
    config = Mock()
    config.request_timeout_seconds = 10
    config.seed = 123
    config.weights = Mock()

    raw = [{"instruction": "i", "output": "o"}]
    validated = [{"input": "x"}]
    processed = (["sample"], [])
    split = Mock(name="split")

    with (
        patch(
            "alpacax.pipelines.instruction.runner.fetch_raw", return_value=Success(raw)
        ) as fetch_raw,
        patch(
            "alpacax.pipelines.instruction.runner.validate_raw_dataset",
            return_value=Success(validated),
        ) as validate_raw_dataset,
        patch(
            "alpacax.pipelines.instruction.runner.process_instructions",
            return_value=Mock(return_value=Success(processed)),
        ) as process_instructions,
        patch(
            "alpacax.pipelines.instruction.runner.shuffle_split",
            return_value=Mock(return_value=Success(split)),
        ) as shuffle_split,
    ):
        result = build_dataset("https://example.com/data.json", tokenizer, config)

    assert result == Success(split)

    fetch_raw.assert_called_once_with(
        "https://example.com/data.json",
        timeout_seconds=10,
    )
    validate_raw_dataset.assert_called_once_with(raw)
    process_instructions.assert_called_once_with(tokenizer, config)
    shuffle_split.assert_called_once_with(config)


def test_build_dataset_short_circuits_on_fetch_failure():
    tokenizer = Mock()
    config = Mock()
    config.request_timeout_seconds = 10

    error = RuntimeError("fetch failed")

    process_fn = Mock(return_value=Success(([], [])))
    shuffle_fn = Mock(return_value=Success(Mock(name="split")))

    with (
        patch(
            "alpacax.pipelines.instruction.runner.fetch_raw",
            return_value=Failure(error),
        ),
        patch("alpacax.pipelines.instruction.runner.validate_raw_dataset") as validate_raw_dataset,
        patch(
            "alpacax.pipelines.instruction.runner.process_instructions",
            return_value=process_fn,
        ) as process_instructions,
        patch(
            "alpacax.pipelines.instruction.runner.shuffle_split",
            return_value=shuffle_fn,
        ) as shuffle_split,
    ):
        result = build_dataset("https://example.com/data.json", tokenizer, config)

    assert result == Failure(error)

    validate_raw_dataset.assert_not_called()

    process_instructions.assert_called_once_with(tokenizer, config)
    shuffle_split.assert_called_once_with(config)

    process_fn.assert_not_called()
    shuffle_fn.assert_not_called()
