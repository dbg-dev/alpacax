from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import optax
from flax import nnx
from jax import Array

# Loss functions only require a forward pass that produces logits. Keep NNX
# Module compatibility for the typed NNX training-step factory.
type LogitsModel = nnx.Module | Callable[[Array], Array]


def cross_entropy_loss(
    model: LogitsModel,
    x: Array,
    y: Array,
) -> tuple[Array, Array]:
    """Return `(loss, logits)` for integer-label cross entropy.

    This is intentionally small and generic. For instruction tuning / causal LM
    training, prefer `masked_cross_entropy_loss` so ignored labels such as -100
    do not contribute to the loss.
    """
    logits = model(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    return loss, logits


def masked_cross_entropy_loss(
    model: LogitsModel,
    x: Array,
    y: Array,
    *,
    ignore_index: int = -100,
) -> tuple[Array, Array]:
    """Return `(loss, logits)` while ignoring labels equal to `ignore_index`."""
    logits = model(x)
    per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits,
        jnp.maximum(y, 0),
    )
    mask = y != ignore_index
    denominator = jnp.maximum(mask.sum(), 1)
    loss = (per_token_loss * mask).sum() / denominator
    return loss, logits
