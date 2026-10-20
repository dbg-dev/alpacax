# Research examples and integration checks

These examples demonstrate that alpacax's data, training, logging, persistence
and analysis components can work together. They are **integration exercises**,
not a reproduction of published reward-model scaling laws.

Reference: Gao, Schulman & Hilton, [*Scaling Laws for Reward Model
Overoptimization* (arXiv:2210.10760)](https://arxiv.org/abs/2210.10760).

## 1. Synthetic proxy reward sanity check

**Use this first:** no pretrained model download or credentials needed.

```sh
uv sync --extra train
uv run --extra train python -m alpacax.examples.gao_proxy_sanity \
    --seed 42 --output-dir runs/gao-sanity-42
```

It trains a separate linear Bradley–Terry proxy with the generic NNX
trainer, evaluates held-out preferences, checks Orbax restoration and
sweeps best-of-N over deliberately distribution-shifted, **synthetic**
candidates. The setup is constructed to show how a proxy can prefer
candidates that score poorly under the independent gold function.

Outputs: `best_of_n.csv`, `history.json`, `summary.json`, and
`checkpoints/`. Use a **fresh output directory** for each run.

The measured seed-42 gold curve peaks at `N=8` and subsequently
declines while learned proxy scores increase. That is a check of the
illustrative implementation, **not** evidence for Gao et al.'s fitted
scaling law. See [synthetic experiment methodology and results](gao-proxy-study.md).

## 2. Pretrained GPT-2 124M candidate / text-proxy pilot

The next exercise ran the *real* `openai-community/gpt2` checkpoint
through alpacax's strict HF-to-NNX mapper and frozen generator,
then trained an independent, **TF-IDF-based** Bradley–Terry proxy on
cached continuations.

```sh
# Download / map pretrained weights and cache responses.
uv run --extra train --extra huggingface --with 'transformers>=4.57,<5' \
  python -m alpacax.examples.gao_gpt2_candidates \
    --prompts-path docs/gao-prompts.example.jsonl \
    --output-path runs/gao-gpt2/candidates.jsonl \
    --seed 42 --candidates-per-prompt 8 --max-new-tokens 20

# Train the independent text proxy, save and restore state, evaluate BoN.
uv run --extra train python -m alpacax.examples.gao_text_best_of_n \
  --cache-path runs/gao-gpt2/candidates.jsonl \
  --output-dir runs/gao-text-42
```

The example requires network access to fetch the public model weights the
first time; use `uv sync --extra train --extra huggingface` beforehand.
The command writes a cache once and refuses to overwrite it.

**Outcome (seed 42, 20 October 2026):**

- **64 pretrained GPT-2 completions** were generated: 8 prompts ×
  8 candidates, with disjoint train/validation/test prompt IDs.
- The independent proxy trained, was validated, and was saved/restored
  with Orbax; the real-weight end-to-end integration passed.
- Held-out preference accuracy was **56.25%** on the small validation set.
- Crucially, **each of the two held-out test prompts had constant gold
  labels**: one 8/8 positive, one 0/8. Therefore the gold, random and
  oracle curves all remained flat at 0.5 for `N=1,2,4,8`. No meaningful
  reward-overoptimization inference can be drawn.
- The binary exact-phrase evaluator tests token presence, **not**
  semantic quality or human preferences.

The [completed CI run](https://github.com/dbg-dev/alpacax/actions/runs/39421817048)
and [experiment record](https://github.com/dbg-dev/alpacax/pull/16)
capture the actual integration result. The temporary experiment PR was
closed **without merging**; the reusable example code is already on
`main`. For sampling details, caveats and larger-candidate instructions,
see [frozen GPT-2 candidate guide](gao-gpt2-candidates.md).

## 3. Inspect the saved results with marimo

```sh
uv run --extra train --with marimo --with matplotlib \
    marimo edit notebooks/gao_best_of_n.py
```

The notebook **reads previously generated files** and never runs model
training or generation when controls change. Set its run-directory input
to `runs/gao-sanity-42` or `runs/gao-text-42`; use the `N` or
theoretical `sqrt(KL)` axis, gold/oracle/random controls, length
diagnostic and logged losses. To open as a read-only interactive app,
use `marimo run` instead of `marimo edit`.

## Interpreting the examples

A passing run means the integration path works. It does **not** establish
the statistical validity of a training objective or reward evaluator.
Prefer checking held-out within-prompt gold-label diversity and
independent preference accuracy before interpreting any BoN curve.

See [Architecture and design](architecture.md) for the library boundaries
that these examples exercise.
