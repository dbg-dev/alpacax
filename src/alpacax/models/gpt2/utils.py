from flax.nnx import Module, Param, state  # pyright: ignore[reportUnknownVariableType]
from jax import tree


def count_parameters(model: Module) -> int:
    """Count number of parameters in a model"""

    params = state(model, Param)  # pyright: ignore[reportUnknownVariableType]
    leaves = tree.leaves(params)
    return sum(x.size for x in leaves)
