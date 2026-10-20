# pyright: reportPrivateUsage=false, reportPrivateLocalImportUsage=false
# These tests intentionally exercise internal pipeline stages.

from unittest.mock import Mock, patch

from returns.result import Failure, Success

from alpacax.pipelines.instruction.runner import _process_instructions
from alpacax.types.errors import RowError


def test_process_instructions_collects_samples_and_row_errors():
    tokenizer = Mock()
    config = Mock()

    sample_1 = Mock(name="sample_1")
    sample_2 = Mock(name="sample_2")
    error = ValueError("bad row")

    data = [
        {"instruction": "i1", "output": "o1"},
        {"instruction": "bad", "output": "bad"},
        {"instruction": "i2", "output": "o2"},
    ]

    with patch(
        "alpacax.pipelines.instruction.runner.process_instruction",
        side_effect=[
            Success(sample_1),
            Failure(error),
            Success(sample_2),
        ],
    ) as process_instruction:
        samples, errors = _process_instructions(data, tokenizer, config).unwrap()

    assert samples == [sample_1, sample_2]

    assert len(errors) == 1
    assert isinstance(errors[0], RowError)
    assert errors[0].row_index == 1
    assert errors[0].error is error

    assert process_instruction.call_count == 3


def test_process_instructions_returns_failure_when_no_valid_samples():
    tokenizer = Mock()
    config = Mock()

    data = [
        {"instruction": "bad1", "output": "bad1"},
        {"instruction": "bad2", "output": "bad2"},
    ]

    with patch(
        "alpacax.pipelines.instruction.runner.process_instruction",
        side_effect=[
            Failure(ValueError("first bad row")),
            Failure(ValueError("second bad row")),
        ],
    ) as mocked:
        import alpacax.pipelines.instruction.runner as runner

        assert runner.process_instruction is mocked
        result = _process_instructions(data, tokenizer, config)

    error = result.failure()

    assert isinstance(error, ValueError)
    assert "no valid samples after processing" in str(error)
    assert "row 0: first bad row" in str(error)
    assert "row 1: second bad row" in str(error)


def test_process_instructions_annotates_error_with_correct_row_index():
    tokenizer = Mock()
    config = Mock()

    data = [
        {"instruction": "ok", "output": "ok"},
        {"instruction": "bad", "output": "bad"},
    ]

    sample = Mock(name="sample")
    error = RuntimeError("boom")

    with patch(
        "alpacax.pipelines.instruction.runner.process_instruction",
        side_effect=[
            Success(sample),
            Failure(error),
        ],
    ):
        samples, errors = _process_instructions(data, tokenizer, config).unwrap()

    assert samples == [sample]
    assert len(errors) == 1
    assert isinstance(errors[0], RowError)
    assert errors[0].row_index == 1
    assert errors[0].error is error
