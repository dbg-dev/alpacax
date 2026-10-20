import pytest

from alpacax.pipelines.instruction.transforms import (
    format_alpaca,
    pad_to,
    shuffle,
    split_samples,
)
from alpacax.types.models import AlpacaSample, DatasetSplit, Instruction, SplitConfig

##
## padding
##


def test_pad_to():
    inputs = [[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]]
    exp = [[0, 1, 2, 3, 4], [5, 6, 50256, 50256, 50256], [7, 8, 9, 50256, 50256]]
    tests = zip(inputs, exp, strict=True)
    for test, exp in tests:
        assert pad_to(test, length=5, pad_value=50256) == exp


def test_pad_to_error():
    with pytest.raises(ValueError, match="token list length 4 exceeds target 3"):
        pad_to([0, 1, 2, 3], length=3, pad_value=-100)


##
## Test instruction formatting
##


def test_full_instruction():
    test = Instruction(
        instruction="Identify the correct spelling of the following word.",
        input="Occasion",
        output="The correct spelling is 'Occasion'.",
    )
    exp = AlpacaSample(
        prompt="""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Identify the correct spelling of the following word.

### Input:
Occasion

""",
        response="""### Response:
The correct spelling is 'Occasion'.""",
    )

    assert format_alpaca(test) == exp


def test_no_input_instruction():
    test = Instruction(
        instruction="What is an antonym of 'complicated'?",
        input="",
        output="An antonym of 'complicated' is 'simple'.",
    )

    exp = AlpacaSample(
        prompt="""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is an antonym of 'complicated'?

""",
        response="""### Response:
An antonym of 'complicated' is 'simple'.""",
    )

    assert format_alpaca(test) == exp


##
## shuffle
##


def test_shuffle():
    ls = list(range(5))
    assert shuffle(ls, seed=4) == [3, 4, 0, 2, 1]


##
## Splitting
##


def test_split_samples():
    ls = list(range(10))
    act: DatasetSplit[int, object] = split_samples(
        ls, skipped=None, weights=SplitConfig(train=0.5, validation=0.4, test=0.1)
    )

    assert act.train == [0, 1, 2, 3, 4]
    assert act.validation == [5, 6, 7, 8]
    assert act.test == [9]
    assert act.skipped is None


def test_split_samples_accounts_for_all_items():
    """No items should be lost or duplicated regardless of n."""
    weights = SplitConfig(train=0.8, validation=0.1, test=0.1)
    for n in range(1, 20):
        samples = list(range(n))
        result: DatasetSplit[int, object] = split_samples(samples, skipped=None, weights=weights)
        assert len(result.train) + len(result.validation) + len(result.test) == n
