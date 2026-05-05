from flax.nnx import Module, Param, state
from jax import tree


def count_parameters(model: Module) -> int:
    """Count number of parameters in a model"""

    params = state(model, Param)
    leaves = tree.leaves(params)
    return sum(x.value.size if hasattr(x, "value") else x.size for x in leaves)
