from collections.abc import Iterable

import pytest

from alpacax.trainer.contracts import HookEntry, Runner, TrainPhase
from alpacax.trainer.core import EmptyDataloaderError, run


class FakeMetric:
    def __init__(self) -> None:
        self.count: int = 0

    def update(self, *, loss: object, logits: object = None, labels: object = None) -> None:
        del loss, logits, labels
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
        self.mode: str = mode

    def train(self) -> None:
        self.mode = "train"

    def eval(self) -> None:
        self.mode = "eval"


def make_train_phase(
    loader: Iterable[tuple[object, object]], model: FakeModel | None = None
) -> TrainPhase[FakeModel, FakeOptimizer, FakeMetric, object, object]:
    metric = FakeMetric()

    def step(model: FakeModel, optimizer: FakeOptimizer, metrics: FakeMetric, x: object, y: object) -> None:
        del model, optimizer
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

    # Only two completed updates, numbered 1 and 2.
    assert train.metrics.count == 2


def test_run_calls_hooks_after_each_step() -> None:
    train = make_train_phase(loader=[("x0", "y0"), ("x1", "y1"), ("x2", "y2")])
    seen_steps: list[int] = []
    runner = Runner(
        train=train,
        hooks=(HookEntry(lambda ctx: seen_steps.append(ctx.step), every=1, lead_only=False),),
    )

    run(runner)

    assert seen_steps == [1, 2, 3]


def test_run_puts_train_model_in_train_mode_for_every_step() -> None:
    # Model arrives in eval mode, e.g. left over from an earlier eval pass.
    model = FakeModel(mode="eval")
    modes_seen: list[str] = []

    train = make_train_phase(loader=[("x0", "y0"), ("x1", "y1")], model=model)
    inner_step = train.step

    def recording_step(model: FakeModel, optimizer: FakeOptimizer, metrics: FakeMetric, x: object, y: object) -> None:
        modes_seen.append(model.mode)
        inner_step(model, optimizer, metrics, x, y)

    train.step = recording_step

    run(Runner(train=train))

    assert modes_seen == ["train", "train"]


@pytest.mark.parametrize(
    ("total_steps", "expected"),
    [
        (1, [1]),
        (99, [99]),
        (100, [100]),
        (101, [100, 101]),
        (200, [100, 200]),
        (201, [100, 200, 201]),
        (250, [100, 200, 250]),
    ],
)
def test_hook_every_100_runs_on_completed_update_and_final_step(
    total_steps: int, expected: list[int]
) -> None:
    seen: list[int] = []
    runner = Runner(
        train=make_train_phase(loader=((i, i) for i in range(total_steps))),
        hooks=(HookEntry(lambda ctx: seen.append(ctx.step), every=100, lead_only=False),),
    )

    run(runner)

    assert seen == expected


def test_hook_interval_never_runs_at_step_zero() -> None:
    entry = HookEntry(lambda ctx: None, every=100)

    assert not entry.should_run(0)
    assert not entry.should_run(99)
    assert entry.should_run(100)


def test_final_hook_can_be_disabled() -> None:
    seen: list[int] = []
    runner = Runner(
        train=make_train_phase(loader=[(i, i) for i in range(3)]),
        hooks=(
            HookEntry(
                lambda ctx: seen.append(ctx.step),
                every=2,
                lead_only=False,
                on_final=False,
            ),
        ),
    )

    run(runner)

    assert seen == [2]


def test_final_hook_runs_at_max_steps_without_duplicate() -> None:
    seen: list[int] = []
    runner = Runner(
        train=make_train_phase(loader=((i, i) for i in range(210))),
        hooks=(HookEntry(lambda ctx: seen.append(ctx.step), every=100, lead_only=False),),
        max_steps=201,
    )

    run(runner)

    assert seen == [100, 200, 201]


def test_max_steps_does_not_consume_an_extra_batch() -> None:
    yielded: list[int] = []

    def loader() -> Iterable[tuple[int, int]]:
        for i in range(5):
            yielded.append(i)
            yield i, i

    train = make_train_phase(loader=loader())
    run(Runner(train=train, max_steps=2))

    assert train.metrics.count == 2
    assert yielded == [0, 1]


@pytest.mark.parametrize("max_steps", [0, -1])
def test_run_rejects_non_positive_max_steps(max_steps: int) -> None:
    runner = Runner(train=make_train_phase(loader=[(1, 1)]), max_steps=max_steps)

    with pytest.raises(ValueError, match="max_steps must be positive"):
        run(runner)


def test_run_with_short_loader_uses_actual_final_step() -> None:
    seen: list[int] = []
    runner = Runner(
        train=make_train_phase(loader=[(1, 1), (2, 2)]),
        hooks=(HookEntry(lambda ctx: seen.append(ctx.step), every=10, lead_only=False),),
        max_steps=100,
    )

    run(runner)

    assert seen == [2]


def test_resumed_runner_uses_global_step_for_hooks() -> None:
    seen: list[int] = []
    train = make_train_phase(loader=[(1, 1), (2, 2), (3, 3)])
    runner = Runner(
        train=train,
        hooks=(HookEntry(lambda ctx: seen.append(ctx.step), every=100, lead_only=False),),
        start_step=199,
        max_steps=2,
    )

    run(runner)

    assert seen == [200, 201]
    assert train.metrics.count == 2


def test_run_rejects_negative_resume_step() -> None:
    runner = Runner(train=make_train_phase(loader=[(1, 1)]), start_step=-1)

    with pytest.raises(ValueError, match="start_step must be non-negative"):
        run(runner)
