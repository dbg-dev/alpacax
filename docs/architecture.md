# Architecture and design

alpacax is a **small, composable toolkit for JAX/Flax NNX experiments**.
Rather than owning a complete model-training application, it provides typed
contracts and small factories that compose the caller's model, data, optimizer,
loss and metrics into a reproducible experiment.

The design separates three concerns:

1. **Preparation:** obtain and validate data in a backend-light package.
2. **Execution:** apply a model-specific loss and Optax optimizer to batches
   using a generic NNX training loop.
3. **Effects and inspection:** evaluate, log, save/restore state, import
   pretrained parameters, and analyze experimental results.

The core trainer does **not** depend on Alpaca instruction records or GPT-2.

## Layers

| Layer | Code | Responsibilities |
|---|---|---|
| Instruction preparation | `pipelines/instruction/`, `types/` | Network fetch, dataset schema validation, tokenization, prompt-masked shifted labels, deterministic splits, partial row-level failures |
| Batch adapters | `trainer/data.py` | In-memory, re-iterable `(x, y)` JAX arrays from padded or compact instruction samples |
| Training contracts | `trainer/contracts.py` | Typed `TrainPhase`, `EvalPhase`, `Runner`, `HookEntry` and logging/checkpoint protocols |
| Step factories | `trainer/steps.py`, `trainer/factories.py` | `nnx.jit` train/eval steps, `nnx.value_and_grad`, Optax update, metrics; assemble the phases |
| Runner and hooks | `trainer/core.py`, `trainer/hooks.py` | Iterate batches, count completed updates, dispatch periodic and final hooks for evaluation, logs and persistence |
| Persistence | `trainer/checkpoints.py` | Orbax save/restore of model/optimizer state and completed global step |
| Model and weight imports | `models/gpt2/`, `params/mapping.py`, `integrations/huggingface.py` | Reference NNX GPT-2, explicit parameter mapping rules, key/shape validation and optional torch conversion |
| Experiment drivers | `examples/`, `notebooks/` | Construct tasks and run studies using the above components; plot saved results with marimo |

`src/alpacax/` is the package root; paths above are relative to it unless
otherwise specified.

## Execution and data flow

```text
  source URL + tokenizer
          |
  build_dataset() / build_unpadded_dataset()
          |
  DatasetSplit[PaddedSample or PreparedSample, DatasetError]
          |
  PaddedBatchLoader / BucketedBatchLoader        other (x, y) iterable
                      \                             /
                       +---- iterable[(x, y)] ----+
                                      |
           caller supplies NNX model, Optax optimizer,
                    loss_fn and NNX metrics
                                      |
                make_train_phase() / make_eval_phase()
                                      |
                                make_runner()
                                      |
                                   run()
                          completed update steps
                                      |
                    HookEntry(every=..., on_final=...)
                          /           |           \
                 logging_hook     eval_hook    checkpoint_entry
                                                    |
                                               Orbax state

      HF/PyTorch state_dict -- mapping rules --> local NNX model
      (independent of where training batches come from)
```

The diagram shows the *instruction-data route* into the generic trainer;
research examples can supply an entirely different `(x, y)` iterable.
For example, the Gao-inspired sanity check trains on pairs of numeric
candidate features, while the text pilot uses features derived from generated
completions.

### The trainer's contracts

A train loader yields **pairs of JAX arrays**; the concrete meaning and
shapes of `x` and `y` are defined by the experiment.

A custom loss follows the contract:

```python
def loss_fn(model, x, y) -> tuple[Array, Array]:
    # Scalar loss and auxiliary predictions/logits for metrics.
    ...
```

Factories connect that loss to NNX JIT-ed steps, the model and optimizer:

```python
train_phase = make_train_phase(
    model=model,
    optimizer=optimizer,
    loader=train_loader,
    loss_fn=loss_fn,
    metrics=metrics,
)
runner = make_runner(train=train_phase, max_steps=100)
run(runner)
```

Here `model`, `optimizer`, `train_loader`, `loss_fn` and `metrics`
are caller-provided. See [training-data.md](training-data.md) for a complete
instruction-dataset bridge, including imports and masking rules, and
[the research examples](research-examples.md) for executable end-to-end drivers.

The step mutates NNX model/optimizer/metric state. The runner itself owns no
optimizer logic. It invokes a train step for each batch and invokes hooks
**after the completed update**. An optional eval phase can use its own loader
and metrics; the `eval_hook` switches the model into eval mode and restores
train mode.

### Hooks, checkpoints and resume

`HookEntry` schedules effects at completed steps (`every=N`) and, by
default, flushes non-due hooks after the final update. Logging and
evaluation are separate from the train step. `checkpoint_entry()` uses
`lead_only=False`, since Orbax save/restore is collective in multi-host runs.

`TrainerCheckpointer` manages Orbax's asynchronous lifecycle. A compatible
NNX model/optimizer graph and sharding target are required to restore; the
completed step supports continuation. **Dataloader position and external RNG
state are not captured**, so resume is *not* deterministic replay of an
interrupted experiment. See [checkpointing.md](checkpointing.md).

### Optional model and integration boundaries

The library deliberately avoids requiring heavyweight backends merely to
prepare instruction data.

- The **core** depends on Pydantic, requests, returns and tyro.
- The `train` extra brings JAX, Flax NNX, Optax and Orbax.
- The `huggingface` extra enables PyTorch state-dict conversion without
  eagerly importing PyTorch in the core pipeline.
- W&B is loaded only when its logger is constructed; Orbax's default
  checkpoint payload builder is loaded only when configured.

The model mapping code uses explicit source-to-target rules. **Strict mode**
checks key coverage and shape compatibility; it is complemented by separate
GPT-2 output-equivalence tests. Details:
[strict-mapping.md](strict-mapping.md).

## Evidence and boundaries

The [Gao-inspired example suite](research-examples.md) exercised the
following full chain with real pretrained GPT-2 124M: model-state import,
seeded text generation, cached candidates, independent pairwise proxy
training, validation, evaluation hooks, Orbax save/restore and best-of-N
reporting. This establishes an **integration check**, not the validity of a
reward-overoptimization scaling law.

Current limitations:

- **Small-scale/data:** instruction loaders are in-memory; streaming, dataset
  packing, device prefetching and large-scale data pipelines are not provided.
- **Resume:** model, optimizer and completed step restore; dataloader
  position and external RNG state do not.
- **Distribution:** Orbax APIs and hooks follow collective multi-host
  requirements, but full multi-host training is not yet validated.
- **Models:** a reference GPT-2 implementation and explicit mapping rules
  are included; alpacax is not a broad model zoo or a high-throughput
  generation engine. The research candidate sampler uses simple full-context
  inference rather than a KV cache.
- **Research:** reproducible experiments and examples exist; scientifically
  meaningful gold/learned rewards, larger held-out sets and multi-seed
  evaluation remain the experiment author's responsibility.

## Where to go next

- [Training data and batch shape semantics](training-data.md)
- [Checkpoint lifecycle](checkpointing.md)
- [Parameter mapping and HF import](strict-mapping.md)
- [Research examples, outcomes, and marimo](research-examples.md)
- [Development and CI](development.md)
