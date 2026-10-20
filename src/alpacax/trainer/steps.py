from flax.nnx import Metric, Module, Optimizer, jit, value_and_grad  # pyright: ignore[reportUnknownVariableType]
from jax import Array

from .contracts import EvalStep, LossWithAuxFn, TrainStep


def make_train_step[ModelT: Module](
    loss_fn: LossWithAuxFn,
    *,
    donate: bool = False,
) -> TrainStep[ModelT, Optimizer[ModelT], Metric, Array, Array]:
    """Create a jitted NNX train step.

    The returned step mutates the NNX model/optimizer graph and updates the
    provided metric object. It returns `None` deliberately: hooks/loggers read
    metric state at scheduled boundaries.
    """

    @jit(donate_argnums=(1,) if donate else ())
    def train_step(
        model: ModelT,
        optimizer: Optimizer[ModelT],
        metrics: Metric,
        x: Array,
        y: Array,
    ) -> None:
        def _loss_fn(model: ModelT) -> tuple[Array, Array]:
            return loss_fn(model, x, y)

        (loss, logits), grads = value_and_grad(
            _loss_fn,
            has_aux=True,
        )(model)

        optimizer.update(model, grads)  # pyright: ignore[reportUnknownMemberType]

        metrics.update(  # pyright: ignore[reportUnknownMemberType]
            loss=loss,
            logits=logits,
            labels=y,
        )

    return train_step


def make_eval_step[ModelT: Module](
    loss_fn: LossWithAuxFn,
) -> EvalStep[ModelT, Metric, Array, Array]:
    """Create a jitted NNX eval step."""

    @jit
    def eval_step(
        model: ModelT,
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
