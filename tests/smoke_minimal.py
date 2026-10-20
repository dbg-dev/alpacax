"""Run in a fresh `uv sync --no-dev` environment, without pytest or ML extras."""

import importlib.util
import sys

from alpacax import PipelineConfig, PreparedSample, SplitConfig, build_unpadded_dataset
from alpacax.params.mapping import TransformKind, infer_transform
from alpacax.trainer.loggers import ConsoleLogger, WandbLogger

assert PipelineConfig(max_seq_len=8, eos_token_id=0)
assert SplitConfig(train=0.8, validation=0.1, test=0.1)
assert PreparedSample(input_ids=[1], labels=[2])
assert callable(build_unpadded_dataset)
assert infer_transform((2, 3), (3, 2), TransformKind.AUTO) is TransformKind.TRANSPOSE_LAST_TWO
assert ConsoleLogger is not None
assert WandbLogger is not None

for package in ("jax", "flax", "grain", "optax", "orbax", "torch", "wandb"):
    assert importlib.util.find_spec(package) is None, f"Unexpected dependency installed: {package}"
    assert package not in sys.modules, f"Unexpected eager import: {package}"

print("Minimal alpacax installation and imports OK")
