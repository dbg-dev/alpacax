from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable

from .contracts import (
    Checkpointer,
    CheckpointPayload,
    Hook,
    HookContext,
    HookEntry,
    MetricsLogger,
)
from .utils import is_lead_process, to_host_metrics


def run_hooks(
    ctx: HookContext,
    hooks: Iterable[HookEntry],
    *,
    final: bool = False,
) -> None:
    """Dispatch periodic hooks, or flush non-due hooks after the final update.

    ``final=True`` is a second pass at the same completed step; hooks already
    triggered by their regular interval must not run twice.
    """
    lead = is_lead_process()

    for entry in hooks:
        if entry.lead_only and not lead:
            continue

        due = entry.should_run(ctx.step)
        if final:
            if not entry.on_final or due:
                continue
        elif not due:
            continue

        entry.fn(ctx)


def logging_hook(logger: MetricsLogger) -> Hook:
    """Compute, log, and reset training metrics."""

    def _hook(ctx: HookContext) -> None:
        values = ctx.train.metrics.compute()
        logger.log(to_host_metrics(values), step=ctx.step)
        ctx.train.metrics.reset()

    return _hook


def eval_hook(
    logger: MetricsLogger,
    *,
    max_batches: int | None = None,
    warn_empty: bool = True,
) -> Hook:
    """Run the eval phase, log eval metrics, then reset eval metrics.

    The eval model runs in eval mode (dropout off). The eval phase may share its
    model with the train phase, so the train model is always handed back in train
    mode afterwards, even if an eval step raises or the loader is empty.
    """

    def _hook(ctx: HookContext) -> None:
        if ctx.eval is None:
            raise ValueError("eval_hook requires an eval phase")

        try:
            ctx.eval.model.eval()
            ctx.eval.metrics.reset()
            batches_seen = 0

            for index, (x, y) in enumerate(ctx.eval.loader):
                if max_batches is not None and index >= max_batches:
                    break

                ctx.eval.step(
                    ctx.eval.model,
                    ctx.eval.metrics,
                    x,
                    y,
                )
                batches_seen += 1

            if batches_seen == 0:
                if warn_empty:
                    warnings.warn(
                        "eval_hook: eval loader produced no batches; skipping logging",
                        stacklevel=2,
                    )
                return

            values = ctx.eval.metrics.compute()
            logger.log(to_host_metrics(values), step=ctx.step)
            ctx.eval.metrics.reset()
        finally:
            ctx.train.model.train()

    return _hook


def checkpoint_hook(
    checkpointer: Checkpointer[CheckpointPayload],
    *,
    payload_fn: Callable[..., CheckpointPayload] | None = None,
) -> Hook:
    """Build a save hook; register with ``lead_only=False`` for Orbax.

    All participating processes must invoke distributed saves on the same steps.
    Prefer ``checkpoint_entry`` for the correct scheduling configuration.
    """
    # Orbax is loaded only if the default checkpoint payload builder is used.
    if payload_fn is None:
        from .checkpoints import checkpoint_payload

        payload_fn = checkpoint_payload

    def _hook(ctx: HookContext) -> None:
        payload = payload_fn(
            model=ctx.train.model,
            optimizer=ctx.train.optimizer,
            metadata={"step": ctx.step},
        )

        checkpointer.save(
            payload,
            step=ctx.step,
        )

    return _hook


def checkpoint_entry(
    checkpointer: Checkpointer[CheckpointPayload],
    *,
    every: int = 100,
    on_final: bool = True,
    payload_fn: Callable[..., CheckpointPayload] | None = None,
) -> HookEntry:
    """Schedule checkpoint saves on all processes at matching completed steps."""
    return HookEntry(
        fn=checkpoint_hook(checkpointer, payload_fn=payload_fn),
        every=every,
        lead_only=False,
        on_final=on_final,
    )
