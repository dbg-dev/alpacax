from typing import override

from flax.nnx import Dropout, Embed, LayerNorm, Module, Rngs
from flax.nnx.helpers import List
from flax.nnx.nn.linear import Linear
from jax import Array
from jax.numpy import arange

from alpacax.models.gpt2.attention import MultiHeadSelfAttention
from alpacax.models.gpt2.config import (
    ActivationFn,
    DenseMLPConfig,
    GPTConfig,
    TransformerBlockConfig,
)


class MLP(Module):
    """
    Feed-forward network: Linear -> activation -> Linear.
    """

    def __init__(self, config: DenseMLPConfig, rngs: Rngs) -> None:
        self.fc1: Linear = Linear(
            config.d_model, config.hidden_dim, use_bias=config.use_bias, rngs=rngs
        )
        self.activation: ActivationFn = config.activation_fn
        self.fc2: Linear = Linear(
            config.hidden_dim, config.d_model, use_bias=config.use_bias, rngs=rngs
        )

    @override
    def __call__(self, x: Array) -> Array:
        mlp = self.fc1(x)
        mlp = self.activation(mlp)
        mlp = self.fc2(mlp)
        return mlp


class TransformerBlock(Module):
    """
    Transformer block for GPT model.
    """

    def __init__(self, config: TransformerBlockConfig, rngs: Rngs) -> None:
        self.mha_norm: LayerNorm = LayerNorm(
            config.d_model, use_bias=config.attention.use_bias, rngs=rngs, epsilon=1e-5
        )
        self.mha: MultiHeadSelfAttention = MultiHeadSelfAttention(config.attention, rngs=rngs)
        self.mha_residual_dropout: Dropout = Dropout(config.dropout_rate, rngs=rngs)

        self.mlp_norm: LayerNorm = LayerNorm(
            config.d_model, use_bias=config.mlp.use_bias, rngs=rngs, epsilon=1e-5
        )
        self.mlp: MLP = MLP(config.mlp, rngs=rngs)
        self.mlp_residual_dropout: Dropout = Dropout(config.dropout_rate, rngs=rngs)

    @override
    def __call__(self, x: Array) -> Array:
        mha = self.mha_norm(x)
        mha = self.mha(mha)
        x = x + self.mha_residual_dropout(mha)
        mlp = self.mlp_norm(x)
        mlp = self.mlp(mlp)
        x = x + self.mlp_residual_dropout(mlp)
        return x


class GPT2(Module):
    def __init__(self, config: GPTConfig, rngs: Rngs) -> None:
        self.max_seq_len: int = config.max_seq_len
        self.token_embedding: Embed = Embed(config.vocab_size, config.d_model, rngs=rngs)
        self.position_embedding: Embed = Embed(
            config.max_seq_len,
            config.d_model,
            rngs=rngs,
        )
        self.embed_dropout: Dropout = Dropout(config.embedding.dropout_rate, rngs=rngs)
        self.blocks: List[TransformerBlock] = List(
            [TransformerBlock(config.block, rngs) for _ in range(config.n_layers)]
        )
        self.layernorm: LayerNorm = LayerNorm(config.d_model, rngs=rngs, epsilon=1e-5)
        self.tie_embeddings: bool = config.tie_embeddings

        if not self.tie_embeddings:
            self.lm_head: Linear = Linear(
                config.d_model,
                config.vocab_size,
                use_bias=False,
                rngs=rngs,
            )

    @override
    def __call__(self, token_ids: Array) -> Array:
        _, seq_len = token_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError("Token length is greater than max sequence length")

        positions = arange(seq_len)[None, :]

        x = self.token_embedding(token_ids)
        x = x + self.position_embedding(positions)
        x = self.embed_dropout(x)

        for block in self.blocks:
            x = block(x)
        x = self.layernorm(x)

        # weight tying
        if self.tie_embeddings:
            return self.token_embedding.attend(x)

        return self.lm_head(x)
