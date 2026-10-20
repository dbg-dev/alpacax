# Reproducible proxy reward training: a Gao-inspired sanity check

[**Gao, Schulman & Hilton (2023), *Scaling Laws for Reward Model
Overoptimization***](https://proceedings.mlr.press/v202/gao23h.html) examines
what happens when a policy is optimized against an imperfect reward-model proxy.
We plan to start with **best-of-N (BoN)**, *not* PPO.

This small script validates that alpacax can train and checkpoint an **independent
Bradley–Terry proxy reward model**, evaluate it on held-out preferences, and
measure proxy versus gold reward as the strength of BoN selection increases.

## Run it

It needs only the existing `train` optional dependencies. **No model downloads
or credentials are required.** Run from the repository root:

```sh
uv sync --extra train
uv run --extra train python -m alpacax.examples.gao_proxy_sanity \
    --seed 42 \
    --output-dir runs/gao-sanity-42
```

Choose a **new output directory** for each run. The example refuses to overwrite
an existing Orbax checkpoint. Use `--help` to see parameters such as
`--train-pairs`, `--validation-pairs`, `--prompts` and `--epochs`.
The default uses 2,048 preference pairs, 256 validation pairs, 128 held-out
candidate pools, and 12 epochs; the model contains just two linear weights.

Output:

- `best_of_n.csv`: BoN sweep at `N = 1, 2, 4, 8, 16, 32, 64, 128`.
- `history.json`: training and validation loss logged once per epoch.
- `summary.json`: seed/config, held-out preference accuracy, restoration check.
- `checkpoints/`: real Orbax model, optimizer and completed-step checkpoint.

Each run checks restored model predictions against the trained model before
writing results. It fails rather than silently accepting a broken restore.

## What is being trained?

Each synthetic candidate has **two numbers**: `benefit` and `length`. A
*fixed, deterministic* gold evaluator computes

\[
r_{\mathrm{gold}}(b,\ell)
= b + 0.7\ell - 2\max(\ell-1,0)^2.
\]

Gold-labelled comparison pairs have `length` in `[0, 0.8)`, where longer
candidates are beneficial. The proxy is a distinct, **linear** NNX model
trained by a Bradley–Terry loss to predict the better candidate in each pair.
Training and validation pairs are generated from separate RNG streams.

Held-out candidate pools instead have a longer-tailed length distribution. The
proxy has not seen the region in which long candidates incur a nonlinear gold
penalty. This *deliberate train/evaluation distribution shift* makes it easy to
inspect proxy overoptimization without waiting for language-model generation.

We create **one fixed, held-out 128-candidate pool per synthetic prompt**, score
it with the trained proxy, and select the best of the first `N` candidates.
The same pool supports every `N`, making comparisons paired rather than
confounded by fresh random sampling. We report:

| CSV field | Meaning |
|---|---|
| `n`, `sqrt_kl` | BoN candidate count and analytical `sqrt(KL)` |
| `proxy_mean` | Mean learned reward of proxy-selected candidates |
| `gold_mean` | Mean gold reward **of those same selected candidates** |
| `random_gold_mean` | Random-choice control from the same candidates |
| `oracle_gold_mean` | Upper bound from selecting by gold reward |
| `length_only_gold_mean` | Control selecting longest candidate |
| `selected_length_mean` | Diagnostic for selection exploiting length |

We use the idealized BoN formula from the paper:

\[
D_{\mathrm{KL}}(\pi_{\mathrm{BoN}}\|\pi_0)
= \log N - \frac{N-1}{N}.
\]

This is a theoretical BoN reference axis, **not an empirical KL estimate**
from an LLM policy. Plot `gold_mean` and `proxy_mean` against `sqrt_kl`
or `n`, alongside random and gold-oracle controls.

## What this establishes—and what it does not

**Established:** the alpacax generic `(x, y)` trainer supports preference
pairs (not just token labels); a custom pairwise loss works with its JIT/NNX
step factory; evaluation hooks record held-out loss; Orbax saves and restores
model/optimizer state; deterministic cached-candidate BoN and controls produce
structured results. These are relevant research-engineering integration tests.

**Not established:** the actual Gao et al. scaling laws, GPT-2 quality, a
trained text reward model, real prompts, SFT, or a natural reward-hacking
effect. The gold function and distribution shift here are *deliberately
constructed* to illustrate a failure mode. We do not fit the paper's scaling
coefficients or compare quantitative results to it.

## From this sanity check to our small research exercise

The planned research study uses **frozen pretrained GPT-2 124M** as the
*generator*, separately trains a proxy reward model, and starts with BoN before
considering PPO. The progression is:

1. **Replace the synthetic candidate pool with cached GPT-2 completions** for
   held-out instruction prompts. Cache prompt IDs, token IDs/text, sampling
   settings, and seeds. Keep training and evaluation prompts disjoint.
2. **Define a deterministic gold evaluator** and build preference comparisons
   on training prompts only. Keep this evaluator separate from the learned
   Bradley–Terry proxy; never train the proxy on held-out candidates.
3. **Replace two numeric features with a text-based reward model**, initially
   modest in size. The loss and alpacax training/evaluation/checkpoint boundary
   can remain the same, but tokenization, score extraction, and batch handling
   will change.
4. Run the `1, 2, 4, …, 128` BoN sweep with **random, gold-oracle and
   length-only** controls; record proxy score, gold score, preference accuracy
   and selected length. Repeat across multiple seeds and proxy-data sizes.
5. Only once the pipeline is validated, investigate scaling-law fits or a PPO
   follow-up.

This separation lets us debug the experimental harness *before* coupling
research outcomes to GPT-2 generation or reward-model architecture.

For instruction SFT inputs and bucketed batches, see
[training-data.md](training-data.md); for checkpoint lifecycle semantics,
see [checkpointing.md](checkpointing.md).
