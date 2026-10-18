from typing import override

from flax.nnx import Dropout, Module, Rngs
from flax.nnx.nn.linear import Linear
from jax import Array
from jax.nn import softmax
from jax.numpy import bool_, inf, split, swapaxes, tri, where

from alpacax.models.gpt2.config import AttentionConfig


class MultiHeadSelfAttention(Module):
    def __init__(self, config: AttentionConfig, rngs: Rngs) -> None:
        self.embed_dim: int = config.d_model
        self.num_heads: int = config.n_heads
        self.head_dim: int = config.head_dim
        self.scale: float = config.head_dim**-0.5
        self.causal: bool = config.causal

        self.qkv: Linear = Linear(
            config.d_model,
            config.qkv_dim,
            use_bias=config.use_bias,
            rngs=rngs,
        )
        self.attn_dropout: Dropout = Dropout(rate=config.dropout_rate, rngs=rngs)
        self.out_proj: Linear = Linear(
            config.d_model, config.d_model, use_bias=config.use_bias, rngs=rngs
        )

    def _split_heads(self, x: Array) -> Array:
        B, L, _ = x.shape
        x = x.reshape(B, L, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: Array) -> Array:
        B, _, L, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, L, self.embed_dim)

    @override
    def __call__(self, x: Array) -> Array:
        _, L, _ = x.shape

        qkv = split(self.qkv(x), 3, axis=-1)
        q, k, v = map(self._split_heads, qkv)
        kT = swapaxes(k, -1, -2)
        attn_scores = (q @ kT) * self.scale

        if self.causal:
            mask = ~tri(L, dtype=bool_)
            attn_scores = where(mask, -inf, attn_scores)

        attn_probs = softmax(attn_scores, axis=-1)  # Note: Softmax applied AFTER the masking

        attn_probs = self.attn_dropout(attn_probs)
        z = attn_probs @ v
        z = self._merge_heads(z)

        return self.out_proj(z)
