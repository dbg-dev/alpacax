"""Bridge instruction pipeline outputs to the trainer's (x, y) batch contract.

This adapter belongs to the optional training integration, not the core
instruction pipeline: importing the core package does not require JAX/NumPy.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from jax import Array

from alpacax.types.models import PaddedSample, PreparedSample


@dataclass(frozen=True, slots=True)
class PaddedBatchLoader:
    """Re-iterable batches of pre-padded samples for the generic trainer.

    Uses the pipeline's already shifted input_ids and labels. The attention
    mask is deliberately not passed: causal right-padding cannot affect earlier
    tokens, and padding/prompt labels are excluded by masked_cross_entropy_loss.

    Shuffling is deterministic for a given seed on each iteration. This is a
    simple in-memory bridge, not a streaming, prefetching, or packing loader.
    """

    samples: Sequence[PaddedSample]
    batch_size: int
    shuffle: bool = False
    seed: int = 42
    drop_last: bool = False

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def __len__(self) -> int:
        count = len(self.samples)
        if self.drop_last:
            return count // self.batch_size
        return (count + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[Array, Array]]:
        order = list(range(len(self.samples)))
        if self.shuffle:
            random.Random(self.seed).shuffle(order)

        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            if self.drop_last and len(indices) < self.batch_size:
                break

            batch = [self.samples[index] for index in indices]
            expected_length = len(batch[0].input_ids)
            for sample in batch:
                if not (
                    len(sample.input_ids)
                    == len(sample.labels)
                    == len(sample.attention_mask)
                    == expected_length
                ):
                    raise ValueError(
                        "Each padded sample must have equally sized input_ids, "
                        "labels, and attention_mask, and batch sequence lengths must match"
                    )

            inputs = np.asarray([sample.input_ids for sample in batch], dtype=np.int32)
            labels = np.asarray([sample.labels for sample in batch], dtype=np.int32)
            yield jnp.asarray(inputs), jnp.asarray(labels)


@dataclass(frozen=True, slots=True)
class BucketedBatchLoader:
    """Re-iterable, length-bucketed batches of unpadded instruction records.

    Each record is assigned to the smallest configured bucket that fits its
    *shifted input length*. Only the batch collator pads it, using pad_token for
    inputs and ignore_index for labels. The causal LM must use right padding
    and masked_cross_entropy_loss for this two-array (x, y) interface.

    Shuffling is deterministic across iterations with the same seed; to change
    ordering between epochs, construct a loader with a different seed.
    """

    samples: Sequence[PreparedSample]
    batch_size: int
    bucket_lengths: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    pad_token: int = 0
    ignore_index: int = -100
    shuffle: bool = False
    seed: int = 42
    drop_last: bool = False
    _groups: tuple[tuple[int, tuple[int, ...]], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if (
            not self.bucket_lengths
            or any(length <= 0 for length in self.bucket_lengths)
            or self.bucket_lengths != tuple(sorted(set(self.bucket_lengths)))
        ):
            raise ValueError(
                "bucket_lengths must be nonempty, strictly increasing positive integers"
            )

        groups: dict[int, list[int]] = {length: [] for length in self.bucket_lengths}
        for index, sample in enumerate(self.samples):
            length = len(sample.input_ids)
            if length == 0 or len(sample.labels) != length:
                raise ValueError(
                    "Prepared samples must contain equal-length, nonempty inputs and labels"
                )
            bucket = next((size for size in self.bucket_lengths if length <= size), None)
            if bucket is None:
                raise ValueError(
                    f"Sample length {length} exceeds largest bucket {self.bucket_lengths[-1]}"
                )
            groups[bucket].append(index)

        object.__setattr__(
            self, "_groups", tuple((size, tuple(indices)) for size, indices in groups.items())
        )

    def __len__(self) -> int:
        return sum(
            len(indices) // self.batch_size
            if self.drop_last
            else (len(indices) + self.batch_size - 1) // self.batch_size
            for _, indices in self._groups
        )

    def __iter__(self) -> Iterator[tuple[Array, Array]]:
        rng = random.Random(self.seed)
        batches: list[tuple[int, list[int]]] = []

        for bucket, group in self._groups:
            indices = list(group)
            if self.shuffle:
                rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                chosen = indices[start : start + self.batch_size]
                if self.drop_last and len(chosen) < self.batch_size:
                    continue
                batches.append((bucket, chosen))

        if self.shuffle:
            rng.shuffle(batches)

        for bucket, indices in batches:
            inputs = np.full((len(indices), bucket), self.pad_token, dtype=np.int32)
            labels = np.full((len(indices), bucket), self.ignore_index, dtype=np.int32)
            for row, index in enumerate(indices):
                sample = self.samples[index]
                length = len(sample.input_ids)
                inputs[row, :length] = sample.input_ids
                labels[row, :length] = sample.labels

            yield jnp.asarray(inputs), jnp.asarray(labels)
