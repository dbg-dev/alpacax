from flax.nnx import Module, Optimizer, jit, value_and_grad
from jax import Array

from .contracts import EvalStep, LossWithAuxFn, Metric, TrainStep


def make_train_step(
    loss_fn: LossWithAuxFn,
    *,
    donate: bool = False,
) -> TrainStep:
    """Create a jitted NNX train step.

    The returned step mutates the NNX model/optimizer graph and updates the
    provided metric object. It returns `None` deliberately: hooks/loggers read
    metric state at scheduled boundaries.
    """

    @jit(donate_argnums=(1,) if donate else ())
    def train_step(
        model: Module,
        optimizer: Optimizer[Module],
        metrics: Metric,
        x: Array,
        y: Array,
    ) -> None:
        def _loss_fn(model: Module) -> tuple[Array, Array]:
            return loss_fn(model, x, y)

        (loss, logits), grads = value_and_grad(
            _loss_fn,
            has_aux=True,
        )(model)

        optimizer.update(model, grads)

        metrics.update(
            loss=loss,
            logits=logits,
            labels=y,
        )

    return train_step


def make_eval_step(loss_fn: LossWithAuxFn) -> EvalStep:
    """Create a jitted NNX eval step."""

    @jit
    def eval_step(
        model: Module,
        metrics: Metric,
        x: Array,
        y: Array,
    ) -> None:
        loss, logits = loss_fn(model, x, y)

        metrics.update(
            loss=loss,
            logits=logits,
            labels=y,
        )

    return eval_step
