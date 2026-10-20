# alpacax

**Composable training infrastructure for JAX and Flax NNX experiments.**

alpacax provides a small set of building blocks for preparing instruction data,
training and evaluating NNX models, importing pretrained weights, and checking
that experimental results can be reproduced. It is intended for **small-scale
ML research and prototyping**, not as a full-featured distributed training
platform.

The original Alpaca-format dataset pipeline remains part of the library; it is
**not a prerequisite** for using the generic trainer.

## What it provides

| Component | Responsibility |
|---|---|
| [Instruction pipeline](docs/training-data.md) | Fetch, validate, format, tokenize, mask, split and prepare Alpaca-format data; retain row-level errors |
| [Training engine](docs/architecture.md) | JIT-compiled NNX train/eval steps driven by an iterable of `(inputs, labels)` batches and a caller-supplied loss |
| [Hooks and checkpoints](docs/checkpointing.md) | Scheduled evaluation, metrics logging, Orbax model/optimizer saves, and state restoration |
| [Model and parameter tools](docs/strict-mapping.md) | GPT-2 reference implementation and explicit Hugging Face/PyTorch-to-NNX parameter mapping |
| [Research examples](docs/research-examples.md) | Reproducible proxy-reward experiments, GPT-2 candidate generation, and a marimo results notebook |

### Architecture at a glance

```text
  Instruction source                         Pretrained HF checkpoint
          |                                            |
    Validate / tokenize                          Mapping rules
    / mask / split                                     |
          |                                       NNX GPT-2
     Batch loaders                                      |
          |                                             |
          +----> (inputs, labels) <---- other data/model experiments
                         |
                   NNX model + loss
                   Optax optimizer
                         |
                 JIT train/eval phases
                         |
                       Runner
                         |
            Hooks: metrics | evaluation | Orbax
```

The **runner coordinates updates**; models, batches and losses remain independent
of the instruction pipeline. Side effects live at hook and integration
boundaries. See [Architecture and design](docs/architecture.md) for the
contracts, modules and current limitations.

## Quick start

Requires **Python 3.12+** and [uv](https://docs.astral.sh/uv/).
From a repository checkout:

```sh
uv sync --extra train

# End-to-end training, evaluation and checkpoint-restore sanity check.
# Use a fresh output directory for each run.
uv run --extra train python -m alpacax.examples.gao_proxy_sanity \
    --seed 42 --output-dir runs/gao-sanity-42
```

To inspect the generated `best_of_n.csv`, losses and run metadata:

```sh
uv run --extra train --with marimo --with matplotlib \
    marimo edit notebooks/gao_best_of_n.py
```

Set the notebook's run directory to `runs/gao-sanity-42`. This first example
uses **synthetic data** and requires no pretrained weights. The
[research examples guide](docs/research-examples.md) also shows the already
validated end-to-end workflow using *real, frozen GPT-2 124M* completions.
Those examples demonstrate integration; they **do not reproduce** Gao et al.'s
reward-model scaling laws.

### Optional dependencies

The minimal package intentionally avoids importing JAX, PyTorch or W&B.

| Install | Enables |
|---|---|
| `pip install alpacax` | Instruction pipeline and CLI |
| `pip install 'alpacax[train]'` | JAX/Flax NNX training, Optax and Orbax |
| `pip install 'alpacax[huggingface]'` | PyTorch/Hugging Face model-state adapter |
| `pip install 'alpacax[train,wandb]'` | Training and optional W&B logging |

For development, use `uv sync --all-extras`. The GPT-2 generation example
additionally uses `transformers` via `uv run --with`, as documented in the
[GPT-2 guide](docs/gao-gpt2-candidates.md).

## Documentation

- [Architecture and design](docs/architecture.md) — layers, public contracts,
  execution flow, integration boundaries, and limitations.
- [Training data](docs/training-data.md) — instruction pipeline, masking,
  padding and re-iterable batch loaders.
- [Checkpointing](docs/checkpointing.md) — Orbax lifecycle, collective saves,
  restore and resume limitations.
- [Strict parameter mapping](docs/strict-mapping.md) — HF-to-NNX key/shape
  validation and model reconstruction.
- [Research examples](docs/research-examples.md) — synthetic Gao-inspired
  experiment, frozen GPT-2 integration and marimo analysis.
- [Installation](docs/installation.md), [development](docs/development.md)
  and [publishing](docs/publishing.md).

## Scope

alpacax currently focuses on **inspectable, reproducible single-device or
small-scale experiments**. Its in-memory instruction loaders are not streaming,
its current checkpoint payload does not resume dataloader/RNG positions, and
multi-host operation has not been validated end to end. See the
[architecture notes](docs/architecture.md) before using it at scale.

---

Project scaffolding originally based on
[simple-modern-uv](https://github.com/jlevy/simple-modern-uv).
