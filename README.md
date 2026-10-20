# alpacax


## Alpaca Dataset Pipeline

Fetches, validates, formats, tokenizes, masks, pads, and splits an Alpaca-format
instruction dataset using a railway-oriented Result pattern for error handling.

Python: 3.12+
Core dependencies: pydantic, requests, returns, tyro

Design goals

- Keep I/O at the boundary and make transformations independently testable.
- Validate remote data shape before running the pipeline.
- Avoid unbounded padding by using a configurable maximum sequence length.
- Support either truncating long samples or skipping them with recorded reasons.
- Mask prompt labels by default for causal-LM instruction fine-tuning.
- Return partial success with skipped row errors whenever possible.


## Optional dependencies

Install only what you use; the core instruction pipeline does not require JAX,
PyTorch or W&B:

```sh
pip install alpacax                   # dataset pipeline + CLI
pip install 'alpacax[train]'          # JAX/Flax training + checkpoints
pip install 'alpacax[huggingface]'    # HF/PyTorch -> Flax adapter
pip install 'alpacax[train,wandb]'    # training + W&B logger
```

With uv in a checkout:

```sh
uv sync --no-dev                  # minimal installation
uv sync --extra train            # training and dev tools
uv sync --all-extras             # full development environment
```

The PyTorch adapter operates on a supplied model's `state_dict()`; alpacax
itself does not eagerly import `torch`. The W&B logger imports `wandb` only
when constructed. The default checkpoint payload builder is loaded only when
`checkpoint_hook()` is configured.

After changing dependency declarations, regenerate and commit `uv.lock` with
`uv lock` so CI and other developers use the updated resolution.

* * *

## Project Docs

For how to install uv and Python, see [installation.md](docs/installation.md).

For development workflows, see [development.md](docs/development.md).

For instructions on publishing to PyPI, see [publishing.md](docs/publishing.md).

For Orbax checkpoint lifecycle and training resume, see [checkpointing.md](docs/checkpointing.md).

For bridging the instruction pipeline to training batches, see [training-data.md](docs/training-data.md).

* * *

*This project was built from
[simple-modern-uv](https://github.com/jlevy/simple-modern-uv).*
