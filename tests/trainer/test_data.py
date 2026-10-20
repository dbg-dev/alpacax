"""Pipeline-to-trainer batching behaviour."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alpacax.trainer.data import PaddedBatchLoader
from alpacax.types.models import PaddedSample


def sample(token: int, *, length: int = 4) -> PaddedSample:
    return PaddedSample(
        input_ids=[token] + [0] * (length - 1),
        labels=[token, -100, -100, -100][:length],
        attention_mask=[1] + [0] * (length - 1),
    )


def test_batches_keep_shifted_labels_and_ignore_index() -> None:
    records = [sample(1), sample(2), sample(3)]
    batches = list(PaddedBatchLoader(records, batch_size=2))

    assert len(batches) == 2
    inputs, labels = batches[0]
    assert inputs.shape == (2, 4)
    assert labels.shape == (2, 4)
    assert inputs.dtype == jnp.int32
    assert labels.dtype == jnp.int32
    assert inputs.tolist() == [[1, 0, 0, 0], [2, 0, 0, 0]]
    assert labels.tolist() == [[1, -100, -100, -100], [2, -100, -100, -100]]
    assert batches[1][0].shape == (1, 4)


def test_drop_last_and_length() -> None:
    records = [sample(1), sample(2), sample(3)]
    loader = PaddedBatchLoader(records, batch_size=2, drop_last=True)

    assert len(loader) == 1
    assert len(list(loader)) == 1
    assert len(PaddedBatchLoader(records, batch_size=2)) == 2


def test_loader_can_be_reiterated_with_deterministic_shuffle() -> None:
    records = [sample(i) for i in range(8)]
    loader = PaddedBatchLoader(records, batch_size=3, shuffle=True, seed=7)

    first = [int(x[0, 0]) for x, _ in loader]
    second = [int(x[0, 0]) for x, _ in loader]
    assert first == second
    assert first != [0, 3, 6]
    assert len(first) == 3


@pytest.mark.parametrize("batch_size", [0, -1])
def test_rejects_invalid_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        PaddedBatchLoader([sample(1)], batch_size=batch_size)


def test_rejects_misaligned_or_ragged_samples() -> None:
    records = [sample(1), sample(2, length=3)]
    with pytest.raises(ValueError, match="sequence lengths must match"):
        list(PaddedBatchLoader(records, batch_size=2))

    malformed = PaddedSample(
        input_ids=[1, 2],
        labels=[2],
        attention_mask=[1, 1],
    )
    with pytest.raises(ValueError, match="equally sized"):
        list(PaddedBatchLoader([malformed], batch_size=1))


def test_empty_loader_is_reiterable_and_has_zero_batches() -> None:
    loader = PaddedBatchLoader([], batch_size=4)
    assert len(loader) == 0
    assert list(loader) == []
    assert list(loader) == []
