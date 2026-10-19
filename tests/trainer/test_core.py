import pytest

from alpacax.trainer.contracts import HookEntry, Runner, TrainPhase
from alpacax.trainer.core import EmptyDataloaderError, run


class FakeMetric:
    def __init__(self) -> None:
        self.count = 0

    def update(self, *, loss, logits=None, labels=None) -> None:
        self.count += 1

    def compute(self) -> dict[str, float]:
        return {"loss": float(self.count)}

    def reset(self) -> None:
        self.count = 0


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


def make_train_phase(loader, model: FakeModel | None = None) -> TrainPhase:
    metric = FakeMetric()

    def step(model, optimizer, metrics, x, y) -> None:
        metrics.update(loss=1.0, logits=x, labels=y)

    return TrainPhase(
        model=model if model is not None else FakeModel(),
        optimizer=FakeOptimizer(),
        loader=loader,
        metrics=metric,
        step=step,
    )


def test_run_raises_for_empty_train_loader() -> None:
    runner = Runner(train=make_train_phase(loader=[]))

    with pytest.raises(EmptyDataloaderError, match="no batches"):
        run(runner)


def test_run_steps_first_batch_and_remaining_batches() -> None:
    train = make_train_phase(loader=[("x0", "y0"), ("x1", "y1"), ("x2", "y2")])
    runner = Runner(train=train)

    run(runner)

    assert train.metrics.count == 3


def test_run_respects_max_steps() -> None:
    train = make_train_phase(loader=[("x0", "y0"), ("x1", "y1"), ("x2", "y2")])
    runner = Runner(train=train, max_steps=2)

    run(runner)

    # step 0 and step 1 run; loop stops before step 2
    assert train.metrics.count == 2


def test_run_calls_hooks_after_each_step() -> None:
    train = make_train_phase(loader=[("x0", "y0"), ("x1", "y1"), ("x2", "y2")])
    seen_steps: list[int] = []
    runner = Runner(
        train=train,
        hooks=(HookEntry(lambda ctx: seen_steps.append(ctx.step), every=1, lead_only=False),),
    )

    run(runner)

    assert seen_steps == [0, 1, 2]


def test_run_puts_train_model_in_train_mode_for_every_step() -> None:
    # Model arrives in eval mode, e.g. left over from an earlier eval pass.
    model = FakeModel(mode="eval")
    modes_seen: list[str] = []

    train = make_train_phase(loader=[("x0", "y0"), ("x1", "y1")], model=model)
    inner_step = train.step

    def recording_step(model, optimizer, metrics, x, y) -> None:
        modes_seen.append(model.mode)
        inner_step(model, optimizer, metrics, x, y)

    train.step = recording_step

    run(Runner(train=train))

    assert modes_seen == ["train", "train"]
