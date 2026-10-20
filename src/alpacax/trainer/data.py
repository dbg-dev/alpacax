"""Bridge pre-padded instruction samples to the trainer's (x, y) batch contract.

This adapter belongs to the optional training integration, not the core
instruction pipeline: importing the core package does not require JAX/NumPy.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array

from alpacax.types.models import PaddedSample


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
