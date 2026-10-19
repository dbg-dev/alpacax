from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from alpacax.trainer.checkpoints import CheckpointPayload
from alpacax.trainer.contracts import (
    EvalPhase,
    HookContext,
    HookEntry,
    TrainPhase,
)
from alpacax.trainer.hooks import checkpoint_hook, eval_hook, logging_hook, run_hooks


class FakeMetric:
    def __init__(self) -> None:
        self.updated: list[tuple[float, object, object]] = []
        self.compute_calls: int = 0
        self.reset_calls: int = 0

    def update(self, *, loss: float, logits: object = None, labels: object = None) -> None:
        self.updated.append((loss, logits, labels))

    def compute(self) -> dict[str, float]:
        self.compute_calls += 1
        return {"loss": float(len(self.updated))}

    def reset(self) -> None:
        self.reset_calls += 1
        self.updated.clear()


class FakeLogger:
    def __init__(self) -> None:
        self.records: list[tuple[dict[str, float], int]] = []

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.records.append((metrics, step))

    def close(self) -> None:
        pass


@dataclass
class FakeOptimizer:
    pass


class FakeModel:
    """Stands in for an NNX module: records train()/eval() mode switches."""

    def __init__(self, mode: str = "train") -> None:
        self.mode = mode

    def train(self) -> None:
        self.mode = "train"

    def eval(self) -> None:
        self.mode = "eval"


def make_train_phase(
    *,
    metric: FakeMetric | None = None,
    loader=None,
    model: FakeModel | None = None,
) -> TrainPhase:
    def step(
        model: object,
        optimizer: FakeOptimizer,
        metrics: FakeMetric,
        x: object,
        y: object,
    ) -> None:
        metrics.update(loss=1.0, logits=x, labels=y)

    return TrainPhase(
        model=model if model is not None else FakeModel(),
        optimizer=FakeOptimizer(),
        loader=loader if loader is not None else [(1, 1)],
        metrics=metric or FakeMetric(),
        step=step,
    )


def make_eval_phase(
    *,
    metric: FakeMetric | None = None,
    loader: object = None,
    model: FakeModel | None = None,
    step=None,
) -> EvalPhase:
    def default_step(model: object, metrics: FakeMetric, x, y) -> None:
        metrics.update(loss=1.0, logits=x, labels=y)

    return EvalPhase(
        model=model if model is not None else FakeModel(),
        loader=loader if loader is not None else [(1, 1), (2, 2)],
        metrics=metric or FakeMetric(),
        step=step or default_step,
    )


def test_hook_entry_rejects_non_positive_interval() -> None:
    with pytest.raises(ValueError, match="positive"):
        HookEntry(lambda ctx: None, every=0)


def test_run_hooks_respects_interval() -> None:
    calls: list[int] = []
    ctx = HookContext(step=3, train=make_train_phase())

    run_hooks(
        ctx,
        hooks=(
            HookEntry(lambda ctx: calls.append(ctx.step), every=2, lead_only=False),
            HookEntry(lambda ctx: calls.append(ctx.step), every=3, lead_only=False),
        ),
    )

    assert calls == [3]


def test_logging_hook_computes_logs_and_resets_train_metrics() -> None:
    metric = FakeMetric()
    metric.update(loss=1.0)
    logger = FakeLogger()
    ctx = HookContext(step=7, train=make_train_phase(metric=metric))

    logging_hook(logger)(ctx)

    assert logger.records == [({"loss": 1.0}, 7)]
    assert metric.compute_calls == 1
    assert metric.reset_calls == 1
    assert metric.updated == []


def test_eval_hook_requires_eval_phase() -> None:
    logger = FakeLogger()
    ctx = HookContext(step=1, train=make_train_phase(), eval=None)

    with pytest.raises(ValueError, match="eval phase"):
        eval_hook(logger)(ctx)


def test_eval_hook_runs_eval_phase_logs_and_resets_eval_metrics() -> None:
    eval_metric = FakeMetric()
    logger = FakeLogger()
    ctx = HookContext(
        step=10,
        train=make_train_phase(),
        eval=make_eval_phase(metric=eval_metric, loader=[("x1", "y1"), ("x2", "y2")]),
    )

    eval_hook(logger)(ctx)

    assert logger.records == [({"loss": 2.0}, 10)]
    assert eval_metric.compute_calls == 1
    # once before eval, once after logging
    assert eval_metric.reset_calls == 2
    assert eval_metric.updated == []


def test_eval_hook_warns_and_does_not_log_when_loader_empty() -> None:
    eval_metric = FakeMetric()
    logger = FakeLogger()
    ctx = HookContext(
        step=10,
        train=make_train_phase(),
        eval=make_eval_phase(metric=eval_metric, loader=[]),
    )

    with pytest.warns(UserWarning, match="produced no batches"):
        eval_hook(logger)(ctx)

    assert logger.records == []
    assert eval_metric.compute_calls == 0
    assert eval_metric.reset_calls == 1


def test_eval_hook_runs_steps_in_eval_mode_then_restores_train_mode() -> None:
    model = FakeModel(mode="train")
    modes_seen: list[str] = []

    def recording_step(model: FakeModel, metrics: FakeMetric, x, y) -> None:
        modes_seen.append(model.mode)
        metrics.update(loss=1.0, logits=x, labels=y)

    # Train and eval phases share one model, as in a typical single-model setup.
    ctx = HookContext(
        step=5,
        train=make_train_phase(model=model),
        eval=make_eval_phase(model=model, step=recording_step),
    )

    eval_hook(FakeLogger())(ctx)

    assert modes_seen == ["eval", "eval"]
    assert model.mode == "train"


def test_eval_hook_restores_train_mode_when_eval_step_raises() -> None:
    model = FakeModel(mode="train")

    def failing_step(model: FakeModel, metrics: FakeMetric, x, y) -> None:
        raise RuntimeError("boom")

    ctx = HookContext(
        step=5,
        train=make_train_phase(model=model),
        eval=make_eval_phase(model=model, step=failing_step),
    )

    with pytest.raises(RuntimeError, match="boom"):
        eval_hook(FakeLogger())(ctx)

    assert model.mode == "train"


def test_eval_hook_restores_train_mode_when_loader_is_empty() -> None:
    model = FakeModel(mode="train")
    ctx = HookContext(
        step=5,
        train=make_train_phase(model=model),
        eval=make_eval_phase(model=model, loader=[]),
    )

    with pytest.warns(UserWarning, match="produced no batches"):
        eval_hook(FakeLogger())(ctx)

    assert model.mode == "train"


def test_eval_hook_with_separate_eval_model_leaves_train_model_in_train_mode() -> None:
    train_model = FakeModel(mode="train")
    eval_model = FakeModel(mode="train")
    ctx = HookContext(
        step=5,
        train=make_train_phase(model=train_model),
        eval=make_eval_phase(model=eval_model),
    )

    eval_hook(FakeLogger())(ctx)

    assert train_model.mode == "train"
    assert eval_model.mode == "eval"


def test_checkpoint_hook_calls_checkpointer_with_payload() -> None:
    class FakeCheckpointer:
        def __init__(self) -> None:
            self.calls: list[tuple[CheckpointPayload, int]] = []

        def save(
            self,
            state: CheckpointPayload,
            *,
            step: int,
        ) -> Path:
            self.calls.append((state, step))
            return Path(f"/fake/checkpoints/{step}")

        def restore(self, path: Path | None = None) -> CheckpointPayload:
            raise FileNotFoundError("No checkpoint available")

        def latest(self) -> Path | None:
            return None

        def exists(self) -> bool:
            return False

        def close(self) -> None:
            pass

    train = make_train_phase()

    payload = {
        "model": "fake-model-state",
        "optimizer": "fake-optimizer-state",
        "metadata": {"step": 42},
    }

    def fake_payload_fn(model, optimizer, metadata):
        assert model is train.model
        assert optimizer is train.optimizer
        assert metadata == {"step": 42}
        return payload

    checkpointer = FakeCheckpointer()
    ctx = HookContext(step=42, train=train)

    checkpoint_hook(
        checkpointer,
        payload_fn=fake_payload_fn,
    )(ctx)

    assert checkpointer.calls == [(payload, 42)]
