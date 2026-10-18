from __future__ import annotations

import jax.numpy as jnp

from alpacax.trainer.losses import masked_cross_entropy_loss


class ConstantLogitModel:
    def __call__(self, x):
        batch, seq = x.shape
        logits = jnp.zeros((batch, seq, 3), dtype=jnp.float32)
        # Make class 1 obviously preferred.
        return logits.at[..., 1].set(10.0)


def test_masked_cross_entropy_loss_ignores_minus_100_labels() -> None:
    model = ConstantLogitModel()
    x = jnp.ones((1, 3), dtype=jnp.int32)
    y = jnp.array([[1, -100, 1]], dtype=jnp.int32)

    loss, logits = masked_cross_entropy_loss(model, x, y)

    assert logits.shape == (1, 3, 3)
    assert float(loss) < 0.001
