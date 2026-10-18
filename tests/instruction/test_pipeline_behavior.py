from returns.result import Success

from alpacax.pipelines.instruction.config import PipelineConfig
from alpacax.pipelines.instruction.runner import _process_instructions
from alpacax.pipelines.instruction.stages import _shuffle_split
from alpacax.types.errors import RowError
from alpacax.types.models import SplitConfig


class WordTokenizer:
    def encode(self, text: str) -> list[int]:
        return [len(word) for word in text.split()]


def test_process_instructions_processes_valid_rows_and_reports_invalid_rows():
    data = [
        {"instruction": "Say hello", "input": "politely", "output": "Hello!"},
        {"instruction": "Missing output"},
        {"instruction": "Say bye", "input": "", "output": "Bye!"},
        {"instruction": "Empty output", "output": ""},
    ]

    config = PipelineConfig(
        max_seq_len=64,
        min_seq_len=2,
        add_eos_token=True,
        eos_token_id=99,
        mask_prompt_labels=True,
    )

    result = _process_instructions(
        data,
        tokenizer=WordTokenizer(),
        config=config,
    )

    assert isinstance(result, Success)

    samples, errors = result.unwrap()

    assert len(samples) == 2
    assert len(errors) == 2

    assert [error.row_index for error in errors if isinstance(error, RowError)] == [
        1,
        3,
    ]

    for sample in samples:
        assert len(sample.input_ids) == config.max_seq_len - 1
        assert len(sample.labels) == config.max_seq_len - 1
        assert len(sample.attention_mask) == config.max_seq_len - 1
        assert any(token != config.pad_token for token in sample.input_ids)
        assert any(label != config.ignore_index for label in sample.labels)


def test_processed_samples_are_split_without_dropping_or_duplicating_rows():
    config = PipelineConfig(
        max_seq_len=64,
        min_seq_len=2,
        eos_token_id=99,
    )

    samples, errors = _process_instructions(
        [
            {"instruction": "Task one", "output": "Answer one"},
            {"instruction": "Task two", "output": "Answer two"},
            {"instruction": "Task three", "output": "Answer three"},
            {"instruction": "Task four", "output": "Answer four"},
            {"instruction": "Bad row", "output": ""},
        ],
        tokenizer=WordTokenizer(),
        config=config,
    ).unwrap()

    split = _shuffle_split(
        (samples, errors),
        seed=123,
        weights=SplitConfig(train=0.5, validation=0.25, test=0.25),
    )

    all_split_samples = split.train + split.validation + split.test

    assert len(split.train) == 2
    assert len(split.validation) == 1
    assert len(split.test) == 1

    assert len(all_split_samples) == len(samples)
    assert {id(sample) for sample in all_split_samples} == {id(sample) for sample in samples}

    assert len(split.skipped or []) == 1
