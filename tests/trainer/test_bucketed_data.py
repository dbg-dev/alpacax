"""Fixed-shape bucketed batching for compact instruction samples."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from alpacax.trainer.data import BucketedBatchLoader
from alpacax.types.models import PreparedSample


def example(token: int, length: int) -> PreparedSample:
    return PreparedSample(
        input_ids=[token] + [token + 1] * (length - 1),
        labels=[-7] + [token + 1] * (length - 1),
    )


def test_buckets_pad_only_to_chosen_shape() -> None:
    records = [example(10, 2), example(20, 3), example(30, 6)]
    loader = BucketedBatchLoader(
        records,
        batch_size=2,
        bucket_lengths=(4, 8),
        pad_token=99,
        ignore_index=-7,
    )
    assert len(loader) == 2
    batches = list(loader)
    assert [x.shape for x, _ in batches] == [(2, 4), (1, 8)]
    x, y = batches[0]
    assert x.dtype == jnp.int32
    assert y.dtype == jnp.int32
    assert x.tolist() == [[10, 11, 99, 99], [20, 21, 21, 99]]
    assert y.tolist() == [[-7, 11, -7, -7], [-7, 21, 21, -7]]
    assert batches[1][0].tolist() == [[30, 31, 31, 31, 31, 31, 99, 99]]


def test_bucketing_yields_every_record_once_and_is_reiterable() -> None:
    records = [example(i + 10, (i % 8) + 1) for i in range(17)]
    loader = BucketedBatchLoader(
        records,
        batch_size=3,
        bucket_lengths=(2, 4, 8),
        shuffle=True,
        seed=12,
    )
    first = [int(v) for x, _ in loader for v in x[:, 0].tolist()]
    second = [int(v) for x, _ in loader for v in x[:, 0].tolist()]
    assert first == second
    assert sorted(first) == list(range(10, 27))
    assert len(list(loader)) == len(loader)
    assert all(x.shape[1] in (2, 4, 8) for x, _ in loader)


def test_drop_last_is_applied_per_bucket() -> None:
    records = [example(10, 2), example(11, 3), example(12, 5), example(13, 6)]
    loader = BucketedBatchLoader(records, batch_size=2, bucket_lengths=(4, 8), drop_last=True)
    assert len(loader) == 2
    assert [x.shape for x, _ in loader] == [(2, 4), (2, 8)]

    incomplete = BucketedBatchLoader(
        records[:3], batch_size=2, bucket_lengths=(4, 8), drop_last=True
    )
    assert len(incomplete) == 1
    assert [x.shape for x, _ in incomplete] == [(2, 4)]


def test_empty_bucketed_loader() -> None:
    loader = BucketedBatchLoader([], batch_size=2, bucket_lengths=(4, 8))
    assert len(loader) == 0
    assert list(loader) == []


@pytest.mark.parametrize("batch_size", [0, -1])
def test_rejects_invalid_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        BucketedBatchLoader([example(10, 2)], batch_size=batch_size)


@pytest.mark.parametrize("buckets", [(), (0, 4), (8, 4), (4, 4)])
def test_rejects_invalid_bucket_lengths(buckets: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="bucket_lengths"):
        BucketedBatchLoader([example(10, 2)], batch_size=2, bucket_lengths=buckets)


def test_rejects_ragged_and_oversized_samples() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        BucketedBatchLoader(
            [PreparedSample(input_ids=[1, 2], labels=[2])],
            batch_size=2,
            bucket_lengths=(4,),
        )
    with pytest.raises(ValueError, match="nonempty"):
        BucketedBatchLoader(
            [PreparedSample(input_ids=[], labels=[])],
            batch_size=2,
            bucket_lengths=(4,),
        )
    with pytest.raises(ValueError, match="exceeds largest bucket"):
        BucketedBatchLoader([example(10, 5)], batch_size=2, bucket_lengths=(2, 4))
