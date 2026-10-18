# tests/models/test_config.py

import pytest
from jax.nn import relu
from pydantic import ValidationError

from alpacax.models.gpt2.config import (
    AttentionConfig,
    DenseMLPConfig,
    EmbeddingConfig,
    GPTConfig,
    TransformerBlockConfig,
)


def test_embedding_config_valid():
    config = EmbeddingConfig(
        vocab_size=50_257,
        d_model=768,
        dropout_rate=0.1,
    )

    assert config.vocab_size == 50_257
    assert config.d_model == 768
    assert config.dropout_rate == 0.1


@pytest.mark.parametrize(
    "field,value",
    [
        ("vocab_size", 0),
        ("d_model", 0),
        ("dropout_rate", -0.1),
        ("dropout_rate", 1.0),
    ],
)
def test_embedding_config_invalid(field, value):
    kwargs = {
        "vocab_size": 50_257,
        "d_model": 768,
        "dropout_rate": 0.1,
    }
    kwargs[field] = value

    with pytest.raises(ValidationError):
        EmbeddingConfig(**kwargs)


def test_dense_mlp_config_uses_expansion_factor():
    config = DenseMLPConfig(
        d_model=768,
        expansion_factor=4.0,
    )

    assert config.hidden_dim == 3072


def test_dense_mlp_config_d_hidden_overrides_expansion_factor():
    config = DenseMLPConfig(
        d_model=768,
        expansion_factor=4.0,
        d_hidden=1024,
    )

    assert config.hidden_dim == 1024


def test_dense_mlp_config_custom_activation_fn():
    config = DenseMLPConfig(
        d_model=768,
        activation_fn=relu,
    )

    assert config.activation_fn is relu


def test_dense_mlp_config_rejects_non_integer_hidden_dim():
    with pytest.raises(ValidationError, match="integer hidden dimension"):
        DenseMLPConfig(
            d_model=768,
            expansion_factor=2.3,
        )


def test_attention_config_valid():
    config = AttentionConfig(
        d_model=768,
        n_heads=12,
        dropout_rate=0.1,
    )

    assert config.head_dim == 64
    assert config.qkv_dim == 2304
    assert config.causal is True


def test_attention_config_rejects_invalid_head_split():
    with pytest.raises(ValidationError, match="divisible by n_heads"):
        AttentionConfig(
            d_model=770,
            n_heads=12,
            dropout_rate=0.1,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("d_model", 0),
        ("n_heads", 0),
        ("dropout_rate", -0.1),
        ("dropout_rate", 1.0),
    ],
)
def test_attention_config_invalid_fields(field, value):
    kwargs = {
        "d_model": 768,
        "n_heads": 12,
        "dropout_rate": 0.1,
    }
    kwargs[field] = value

    with pytest.raises(ValidationError):
        AttentionConfig(**kwargs)


def test_transformer_block_config_valid():
    attention = AttentionConfig(
        d_model=768,
        n_heads=12,
        dropout_rate=0.1,
    )
    mlp = DenseMLPConfig(
        d_model=768,
        expansion_factor=4.0,
    )

    config = TransformerBlockConfig(
        attention=attention,
        mlp=mlp,
        dropout_rate=0.1,
    )

    assert config.d_model == 768


def test_transformer_block_config_rejects_mismatched_d_model():
    attention = AttentionConfig(
        d_model=768,
        n_heads=12,
        dropout_rate=0.1,
    )
    mlp = DenseMLPConfig(
        d_model=1024,
        expansion_factor=4.0,
    )

    with pytest.raises(ValidationError, match="attention and mlp d_model must match"):
        TransformerBlockConfig(
            attention=attention,
            mlp=mlp,
            dropout_rate=0.1,
        )


def test_gpt_config_valid():
    embedding = EmbeddingConfig(
        vocab_size=50_257,
        d_model=768,
        dropout_rate=0.1,
    )
    attention = AttentionConfig(
        d_model=768,
        n_heads=12,
        dropout_rate=0.1,
    )
    mlp = DenseMLPConfig(
        d_model=768,
        expansion_factor=4.0,
    )
    block = TransformerBlockConfig(
        attention=attention,
        mlp=mlp,
        dropout_rate=0.1,
    )

    config = GPTConfig(
        embedding=embedding,
        block=block,
        n_layers=12,
        max_seq_len=1024,
        tie_embeddings=True,
    )

    assert config.d_model == 768
    assert config.vocab_size == 50_257
    assert config.mlp_hidden_dim == 3072
    assert config.attention_head_dim == 64


def test_gpt_config_rejects_mismatched_embedding_and_block_d_model():
    embedding = EmbeddingConfig(
        vocab_size=50_257,
        d_model=512,
        dropout_rate=0.1,
    )
    attention = AttentionConfig(
        d_model=768,
        n_heads=12,
        dropout_rate=0.1,
    )
    mlp = DenseMLPConfig(
        d_model=768,
        expansion_factor=4.0,
    )
    block = TransformerBlockConfig(
        attention=attention,
        mlp=mlp,
        dropout_rate=0.1,
    )

    with pytest.raises(ValidationError, match="block and embedding d_model must match"):
        GPTConfig(
            embedding=embedding,
            block=block,
            n_layers=12,
            max_seq_len=1024,
        )


def test_configs_are_frozen():
    config = DenseMLPConfig(d_model=768)

    with pytest.raises(ValidationError):
        config.d_model = 1024


def test_extra_fields_are_forbidden():
    with pytest.raises(ValidationError):
        DenseMLPConfig(
            d_model=768,
            unknown_field=True,
        )
