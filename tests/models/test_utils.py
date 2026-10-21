"""Tests for model utilities."""

from __future__ import annotations

import jax
from flax import nnx

from alpacax.models.gpt2.config import (
    AttentionConfig,
    DenseMLPConfig,
    EmbeddingConfig,
    GPTConfig,
    TransformerBlockConfig,
)
from alpacax.models.gpt2.gpt2 import GPT2
from alpacax.models.gpt2.utils import count_parameters


class _Mixed(nnx.Module):
    """One module with trainable params plus non-param state that must not be counted."""

    def __init__(self, rngs: nnx.Rngs) -> None:
        self.linear: nnx.Linear = nnx.Linear(3, 4, rngs=rngs)  # kernel 3*4 + bias 4 = 16
        # scale 4 + bias 4 = 8 (running stats excluded)
        self.norm: nnx.BatchNorm = nnx.BatchNorm(4, rngs=rngs)
        self.dropout: nnx.Dropout = nnx.Dropout(0.1, rngs=rngs)  # RNG state, excluded


def test_count_parameters_counts_only_trainable_params() -> None:
    assert count_parameters(_Mixed(nnx.Rngs(0))) == 16 + 8


def test_count_parameters_matches_independent_count_on_gpt2() -> None:
    config = GPTConfig(
        embedding=EmbeddingConfig(vocab_size=32, d_model=8, dropout_rate=0.0),
        block=TransformerBlockConfig(
            attention=AttentionConfig(d_model=8, n_heads=2, dropout_rate=0.0),
            mlp=DenseMLPConfig(d_model=8, expansion_factor=2.0),
            dropout_rate=0.0,
        ),
        n_layers=1,
        max_seq_len=16,
    )
    model = GPT2(config, nnx.Rngs(0))

    # A different access path: plain arrays via a pure dict, not Variable objects.
    pure = nnx.to_pure_dict(nnx.state(model, nnx.Param))  # pyright: ignore[reportUnknownMemberType]
    expected = sum(int(array.size) for array in jax.tree.leaves(pure))

    assert count_parameters(model) == expected
    assert expected > 0
