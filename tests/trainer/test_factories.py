from alpacax.trainer.contracts import Runner, TrainPhase
from alpacax.trainer.factories import make_runner


class FakeMetric:
    def update(self, **kwargs) -> None:
        pass

    def compute(self) -> dict[str, float]:
        return {}

    def reset(self) -> None:
        pass


class FakeOptimizer:
    pass


def test_make_runner_normalises_hooks_to_tuple() -> None:
    def train_step(model, optimizer, metrics, x, y) -> None:
        pass

    train = TrainPhase(
        model=object(),
        optimizer=FakeOptimizer(),
        loader=[],
        metrics=FakeMetric(),
        step=train_step,
    )

    runner = make_runner(train=train, hooks=[])

    assert isinstance(runner, Runner)
    assert runner.train is train
    assert runner.hooks == ()
