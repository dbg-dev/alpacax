from alpacax.trainer.contracts import Runner, TrainPhase
from alpacax.trainer.factories import make_runner


class FakeMetric:
    def update(self, **kwargs: object) -> None:
        del kwargs

    def compute(self) -> dict[str, float]:
        return {}

    def reset(self) -> None:
        pass


class FakeOptimizer:
    pass


class FakeModel:
    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass


def test_make_runner_normalises_hooks_to_tuple() -> None:
    def train_step(model: FakeModel, optimizer: FakeOptimizer, metrics: FakeMetric, x: object, y: object) -> None:
        del model, optimizer, metrics, x, y

    train = TrainPhase[FakeModel, FakeOptimizer, FakeMetric, object, object](
        model=FakeModel(),
        optimizer=FakeOptimizer(),
        loader=[],
        metrics=FakeMetric(),
        step=train_step,
    )

    runner = make_runner(train=train, hooks=[])

    assert isinstance(runner, Runner)
    assert runner.train is train
    assert runner.hooks == ()
