# Replacing synthetic candidates with frozen GPT-2 completions

This is the **completed real-model integration exercise** that followed the [Gao synthetic sanity check](gao-proxy-study.md).
Reference: [Gao, Schulman & Hilton (arXiv:2210.10760)](https://arxiv.org/abs/2210.10760).

It uses `alpacax.models.gpt2.GPT2` for actual autoregressive generation. The
pretrained checkpoint is loaded from Hugging Face **into the local Flax NNX
model using the strict GPT-2 parameter-mapping rules**. Model weights remain
frozen and the candidate cache is generated **once**; changing best-of-N does
not generate additional samples.

This is **not** a replication of Gao et al.'s gold or proxy reward models:
the follow-up proxy is an independent **TF-IDF linear Bradley–Terry**
model, and the gold evaluator is a **binary exact-target phrase match**.
This objectively tests a text property, *not* response quality or human
preferences. It can be degenerate on base GPT-2; diagnose that rather than
silently manufacturing preferences.

## 1. Create an independently split prompt file

See [`gao-prompts.example.jsonl`](gao-prompts.example.jsonl). Each JSON line
has exactly these fields:

```json
{"prompt_id":"train-1","split":"train","prompt":"A story began: ","target":"the"}
```

Use disjoint **prompt IDs** for `train`, `validation`, and `test`.
The `target` is the literal, case-insensitive phrase to find in the
**generated continuation only**. A match is a gold score of 1, otherwise 0.
Different train/validation/test prompts are necessary to avoid leakage.

## 2. Generate cached candidates with the pretrained local GPT-2

This stage downloads the public 124M GPT-2 checkpoint and tokenizer on the
first run, requiring network access, enough RAM, and potentially a JAX GPU.
Its small default (8 candidates per prompt; 20 new tokens) is a **smoke pilot**,
not a scaling experiment.

```sh
uv sync --extra train --extra huggingface

uv run --extra train --extra huggingface --with 'transformers>=4.57,<5' \
  python -m alpacax.examples.gao_gpt2_candidates \
    --prompts-path docs/gao-prompts.example.jsonl \
    --output-path runs/gao-gpt2/candidates.jsonl \
    --model-name openai-community/gpt2 \
    --seed 42 --candidates-per-prompt 8 \
    --max-new-tokens 20 --batch-size 4
```

A complete best-of-128 sweep needs **128 per prompt**, not the default 8:

```sh
# Use a fresh output path. This can be substantially more expensive.
uv run --extra train --extra huggingface --with 'transformers>=4.57,<5' \
  python -m alpacax.examples.gao_gpt2_candidates \
    --prompts-path docs/gao-prompts.example.jsonl \
    --output-path runs/gao-gpt2/candidates-128.jsonl \
    --candidates-per-prompt 128
```

The JSONL cache contains prompt ID, split, prompt, gold target, candidate index,
completion text, generated token IDs, model identifier, seed, top-k,
temperature and requested generation length.

**Reproducibility:** sampling uses a per-prompt stable seed and an independent
folded JAX PRNG key for every candidate, regardless of batch size. All
candidate pools are nested at `N=1,2,4,...` and reuse the *same cache*.
A full-context fixed-shape NNX sampler is deliberately simple but slower than
a KV-cache sampler; no speed or scalability claim is intended.

The generator refuses to overwrite an existing cache file.

## 3. Train the independent text proxy and evaluate BoN

```sh
uv run --extra train python -m alpacax.examples.gao_text_best_of_n \
  --cache-path runs/gao-gpt2/candidates.jsonl \
  --output-dir runs/gao-text-42 \
  --seed 42 --max-features 256 --epochs 10
```

The TF-IDF vocabulary and IDF weights are learned **only from train prompt
completions**. Training pairs use only within-prompt candidates with different
binary gold labels. Validation pairs are from separate prompts; held-out test
candidates are **never** fitted or used to construct training preferences.
The proxy optimizes a pairwise Bradley–Terry loss through alpacax's generic
NNX trainer; its own optimizer/checkpoint are separate from the frozen generator.

There must be at least one gold-positive and one gold-negative completion for
some prompt in each of the training and validation splits. Otherwise the
script **stops with a diagnostic**; don't interpret a dataset with all-zero
gold rewards as evidence of reward-model overoptimization. Start by inspecting
the cached completions and choosing an attainable gold property.

Outputs use the **same schema** as the synthetic example:
`best_of_n.csv`, `history.json`, `summary.json`, `checkpoints/`.
A test split with eight candidates supports `N=1,2,4,8`; increasing the
cached candidate count enables longer sweeps.

## Observed pretrained GPT-2 124M integration result

The [20 October 2026 CI run](https://github.com/dbg-dev/alpacax/actions/runs/39421817048)
completed this exact type of workflow with the public pretrained GPT-2 124M
checkpoint: **64 real continuations**, independent text-proxy training,
held-out evaluation, Orbax restoration and best-of-N reporting. All
integration steps succeeded.

This is an **engineering result**, not a reward-overoptimization result:

- Held-out preference accuracy was **56.25%**.
- The two test prompts contained respectively **8/8 positive** and
  **0/8 positive** gold labels; there was no variation *within* either
  prompt.
- Therefore the gold, oracle and random baseline curves all stayed flat
  at **0.5** for `N=1,2,4,8`. The proxy's increasing score did not
  translate into an interpretable gold comparison.
- All sampled continuations reached the 20-token cap, so the selected-length
  diagnostic also lacked variation.

See [research-examples.md](research-examples.md) for the summary and
[experiment PR #16](https://github.com/dbg-dev/alpacax/pull/16) for the run
record (the temporary CI experiment PR was closed, unmerged). No model
weights or output cache are committed to this repository.

## 4. Open the same marimo curves notebook

The [marimo notebook](../notebooks/gao_best_of_n.py) is included in the repository.
From the repository root run:

```sh
uv run --extra train --with marimo --with matplotlib \
  marimo edit notebooks/gao_best_of_n.py
```

Use `runs/gao-text-42` as the notebook's run directory. The notebook only
reads the CSV/JSON results, so changing an `N` or KL plot does not generate
candidates again.

## Experimental interpretation and next gates

- A positive/negative target hit is a **synthetic objective on real text**.
  It is not a gold-quality reward model, and base GPT-2 is not instruction-tuned.
- The proxy is independent and trained with pairwise comparisons; it is not
  sharing any GPT-2 weights.
- With binary gold on very few held-out prompts, the curve can be stepped,
  noisy or flat. A monotone or flat curve does **not** disprove Gao et al.
- Repeat with **two training-pair budgets** and several seeds. Hold generation
  settings and the cached held-out responses fixed when comparing proxy models.
- For a stronger research claim, replace exact-token gold with a validated,
  independent evaluator and consider a stronger separately trained text proxy.
  Only then attempt scaling-law fits; no PPO/DPO in this stage.
