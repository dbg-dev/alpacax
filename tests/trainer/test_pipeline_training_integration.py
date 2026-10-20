"""Train a real NNX model using batches built from the instruction pipeline."""

from __future__ import annotations

from functools import partial
from typing import override

import jax.numpy as jnp
import optax
import pytest
from flax import nnx
from jax import Array
from returns.result import Success

from alpacax.pipelines.instruction.config import PipelineConfig
from alpacax.pipelines.instruction.runner import build_dataset, build_unpadded_dataset
from alpacax.trainer.core import run
from alpacax.trainer.data import BucketedBatchLoader, PaddedBatchLoader
from alpacax.trainer.factories import make_runner, make_train_phase
from alpacax.trainer.losses import masked_cross_entropy_loss
from alpacax.types.models import SplitConfig


class WordTokenizer:
    def encode(self, text: str) -> list[int]:
        # Keep all produced IDs inside the tiny model's vocabulary.
        return [min(len(word), 62) + 1 for word in text.split()]


class TinyLanguageModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.embedding: nnx.Embed = nnx.Embed(64, 8, rngs=rngs)
        self.output: nnx.Linear = nnx.Linear(8, 64, rngs=rngs)

    @override
    def __call__(self, token_ids: Array) -> Array:
        return self.output(self.embedding(token_ids))


def test_pipeline_output_trains_real_nnx_model(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"instruction": f"Say hello {i}", "output": f"Hello number {i}!"} for i in range(4)]

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> list[dict[str, str]]:
            return rows

    def fake_get(url: str, timeout: float) -> Response:
        assert url == "https://example.test/alpaca.json"
        assert timeout > 0
        return Response()

    monkeypatch.setattr("alpacax.pipelines.instruction.pipeio.requests.get", fake_get)
    config = PipelineConfig(
        max_seq_len=64,
        weights=SplitConfig(train=1.0, validation=0.0, test=0.0),
        eos_token_id=2,
        mask_prompt_labels=True,
    )
    result = build_dataset("https://example.test/alpaca.json", WordTokenizer(), config)
    assert isinstance(result, Success)
    records = result.unwrap().train
    assert len(records) == 4
    assert all(-100 in record.labels for record in records)
    assert all(any(label != -100 for label in record.labels) for record in records)

    loader = PaddedBatchLoader(records, batch_size=2, shuffle=True, seed=12)
    assert len(loader) == 2
    x, y = next(iter(loader))
    assert x.shape == y.shape == (2, 63)
    assert jnp.all((y == -100) | ((y >= 0) & (y < 64)))

    model = TinyLanguageModel(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    train = make_train_phase(
        model=model,
        optimizer=optimizer,
        loader=loader,
        metrics=nnx.MultiMetric(loss=nnx.metrics.Average("loss")),
        loss_fn=partial(masked_cross_entropy_loss, ignore_index=config.ignore_index),
    )
    run(make_runner(train=train, max_steps=2))
    assert int(optimizer.step.value) == 2
    # Eval and subsequent epochs can request a fresh iterator.
    assert len(list(loader)) == 2


def test_unpadded_pipeline_trains_real_nnx_model(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "instruction": f"Say hello {i}",
            "output": " ".join(["hello"] * (1 + i * 7)),
        }
        for i in range(4)
    ]

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> list[dict[str, str]]:
            return rows

    monkeypatch.setattr(
        "alpacax.pipelines.instruction.pipeio.requests.get",
        lambda url, timeout: Response(),
    )
    config = PipelineConfig(
        max_seq_len=64,
        weights=SplitConfig(train=1.0, validation=0.0, test=0.0),
        eos_token_id=2,
        mask_prompt_labels=True,
    )
    result = build_unpadded_dataset("https://example.test/alpaca.json", WordTokenizer(), config)
    assert isinstance(result, Success)
    records = result.unwrap().train
    assert len(records) == 4
    assert len({len(record.input_ids) for record in records}) > 1
    assert all(len(record.input_ids) < config.max_seq_len - 1 for record in records)
    assert all(any(label != config.ignore_index for label in record.labels) for record in records)

    loader = BucketedBatchLoader(
        records,
        batch_size=2,
        bucket_lengths=(16, 32, 64),
        pad_token=config.pad_token,
        ignore_index=config.ignore_index,
        shuffle=True,
        seed=config.seed,
    )
    assert sum(x.shape[0] for x, _ in loader) == len(records)
    assert all(x.shape[1] in (16, 32, 64) for x, _ in loader)
    assert len(loader) >= 2

    model = TinyLanguageModel(nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    train = make_train_phase(
        model=model,
        optimizer=optimizer,
        loader=loader,
        metrics=nnx.MultiMetric(loss=nnx.metrics.Average("loss")),
        loss_fn=partial(masked_cross_entropy_loss, ignore_index=config.ignore_index),
    )
    run(make_runner(train=train, max_steps=2))
    assert int(optimizer.step.value) == 2
