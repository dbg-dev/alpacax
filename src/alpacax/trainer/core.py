from itertools import islice

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
    if runner.max_steps is not None and runner.max_steps <= 0:
        raise ValueError("max_steps must be positive")

    runner.train.model.train()
    batches = iter(runner.train.loader)
    if runner.max_steps is not None:
        batches = islice(batches, runner.max_steps)

    last_ctx: HookContext | None = None
    for step, (x, y) in enumerate(batches, start=1):
        runner.train.step(
            runner.train.model,
            runner.train.optimizer,
            runner.train.metrics,
            x,
            y,
        )
        last_ctx = HookContext(step=step, train=runner.train, eval=runner.eval)
        run_hooks(last_ctx, runner.hooks)

    if last_ctx is None:
        raise EmptyDataloaderError("train loader produced no batches")

    # At normal termination, flush hooks whose interval did not land exactly
    # on the last completed update, without duplicate logging/checkpoints.
    run_hooks(last_ctx, runner.hooks, final=True)
