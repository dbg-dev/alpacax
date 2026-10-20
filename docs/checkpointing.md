# Training checkpoint lifecycle

`TrainerCheckpointer` wraps Orbax for Flax NNX model and optimizer state.
Use it as a context manager so outstanding asynchronous writes finish and
resources are released on both normal termination and exceptions.

```python
from alpacax.trainer.checkpoints import TrainerCheckpointer, restore_training_state
from alpacax.trainer.core import run
from alpacax.trainer.factories import make_runner
from alpacax.trainer.hooks import checkpoint_entry

# Construct a compatible model, optimizer, and training phase first.
with TrainerCheckpointer("checkpoints") as checkpointer:
    start_step = (
        restore_training_state(checkpointer, model=model, optimizer=optimizer)
        if checkpointer.exists()
        else 0
    )
    runner = make_runner(
        train=train_phase,
        hooks=(checkpoint_entry(checkpointer, every=100),),
        start_step=start_step,
        max_steps=1000,
    )
    run(runner)
# All outstanding checkpoint writes are complete.
```

`restore_training_state` updates the existing model and optimizer in place
and returns the **last completed global step**. The next update is numbered
`start_step + 1`. `max_steps` counts the updates in *this* invocation, not
the total across earlier runs. Checkpoint entries also save at the last completed
step if it is not a regular checkpoint boundary.

The restore target is derived from the newly constructed model and optimizer,
so their structure, shapes, dtypes, and JAX sharding must be compatible with
the saved checkpoint. For sharded arrays, construct the objects under the
intended mesh before restoring.

## Multi-host use

Orbax checkpoint saves and restores are **collective**. Create and use the
checkpointer on **all participating processes** with the same checkpoint
directory, matching steps, and compatible sharded state. In particular, use
`checkpoint_entry()`, which sets `lead_only=False`; do **not** wrap
`checkpoint_hook()` with the default lead-only `HookEntry` in a multi-host
run. Every host must reach the same checkpoint steps and close its manager.

Logging hooks can remain lead-only. Never use a single checkpoint manager
concurrently from independent jobs, including separate evaluator jobs.

## Resume limitations

The current checkpoint saves model and optimizer state plus the completed
step. It does **not** restore dataloader position, external RNG state, or
other training process state. The caller must provide a suitable resumed data
stream. This supports continued training, but **not** exact deterministic
replay of an interrupted run.
