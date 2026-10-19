from .contracts import HookContext, Runner
from .hooks import run_hooks


class TrainerError(Exception):
    """Base error for trainer-level failures."""


class EmptyDataloaderError(TrainerError):
    """Raised when a training dataloader yields no batches."""


def run(runner: Runner) -> None:
    """Run training.

    The train phase updates model/optimizer/metrics. Hooks own side effects such
    as logging, eval, checkpointing, and metric compute/reset boundaries.

    The train model is put in train mode (dropout active) before the first step.
    ``eval_hook`` switches to eval mode for its pass and hands the train model
    back in train mode afterwards.
    """
    runner.train.model.train()
    iterator = iter(runner.train.loader)

    try:
        first_x, first_y = next(iterator)
    except StopIteration as exc:
        raise EmptyDataloaderError("train loader produced no batches") from exc

    runner.train.step(
        runner.train.model,
        runner.train.optimizer,
        runner.train.metrics,
        first_x,
        first_y,
    )
    run_hooks(
        HookContext(step=0, train=runner.train, eval=runner.eval),
        runner.hooks,
    )

    for step, (x, y) in enumerate(iterator, start=1):
        if runner.max_steps is not None and step >= runner.max_steps:
            break

        runner.train.step(
            runner.train.model,
            runner.train.optimizer,
            runner.train.metrics,
            x,
            y,
        )
        run_hooks(
            HookContext(step=step, train=runner.train, eval=runner.eval),
            runner.hooks,
        )
