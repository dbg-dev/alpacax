"""Train an independent TF-IDF Bradley-Terry proxy from cached GPT-2 text.

Gold is an explicitly chosen *binary exact-target* test, NOT a human-preference
or semantic-quality metric. This is a pilot of Gao's experimental architecture,
not a reproduction of the paper's reward-model scaling-law measurements.

Usage:
    uv run --extra train python -m alpacax.examples.gao_text_best_of_n \
        --cache-path runs/gao-gpt2/candidates.jsonl \
        --output-dir runs/gao-text-42
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import override

import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax import nnx
from jax import Array

from alpacax.examples.gao_proxy_sanity import (
    BEST_OF_N,
    HistoryLogger,
    PairBatches,
    preference_loss,
)
from alpacax.trainer.checkpoints import TrainerCheckpointer, restore_training_state
from alpacax.trainer.contracts import HookEntry
from alpacax.trainer.core import run
from alpacax.trainer.factories import make_eval_phase, make_runner, make_train_phase
from alpacax.trainer.hooks import checkpoint_entry, eval_hook, logging_hook

WORDS = re.compile(r"[\w']+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class Candidate:
    prompt_id: str
    split: str
    prompt: str
    target: str
    candidate_index: int
    completion: str
    token_ids: list[int]


@dataclass(frozen=True, slots=True)
class Settings:
    cache_path: Path = Path("runs/gao-gpt2/candidates.jsonl")
    output_dir: Path = Path("runs/gao-text-42")
    seed: int = 42
    max_features: int = 256
    train_pairs_per_prompt: int = 16
    validation_pairs_per_prompt: int = 8
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.01
    weight_decay: float = 0.01

    def __post_init__(self) -> None:
        if (
            min(
                self.max_features,
                self.train_pairs_per_prompt,
                self.validation_pairs_per_prompt,
                self.batch_size,
                self.epochs,
            )
            < 1
        ):
            raise ValueError(
                "Feature dimension, pair counts, batch size and epochs must be positive"
            )
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid learning rate or regularization strength")


def load_cache(path: Path) -> list[Candidate]:
    result: list[Candidate] = []
    seen: set[tuple[str, int]] = set()
    prompts: dict[str, tuple[str, str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            raw = json.loads(raw_line)
            row = Candidate(**{k: raw[k] for k in Candidate.__dataclass_fields__})
            if row.split not in {"train", "validation", "test"}:
                raise ValueError(f"Line {number}: invalid split")
            if not row.prompt_id or not row.target:
                raise ValueError(f"Line {number}: missing prompt ID or target")
            identity = (row.prompt_id, row.candidate_index)
            if identity in seen:
                raise ValueError(f"Line {number}: duplicate candidate {identity}")
            seen.add(identity)
            descriptor = (row.split, row.prompt, row.target)
            if row.prompt_id in prompts and prompts[row.prompt_id] != descriptor:
                raise ValueError(f"Line {number}: prompt metadata differs for {row.prompt_id}")
            prompts[row.prompt_id] = descriptor
            result.append(row)
    if {r.split for r in result} != {"train", "validation", "test"}:
        raise ValueError("Cache must contain disjoint train, validation and test prompts")
    return result


def gold_binary(completion: str, target: str) -> float:
    """Deterministic target-phrase hit with word boundaries, never semantic grading."""
    pattern = r"(?<!\w)" + re.escape(target.strip()) + r"(?!\w)"
    return float(bool(re.search(pattern, completion, flags=re.IGNORECASE)))


def text_terms(text: str) -> list[str]:
    tokens = WORDS.findall(text.casefold())
    return tokens + [f"{a}__{b}" for a, b in zip(tokens, tokens[1:], strict=False)]


class TfidfFeatures:
    def __init__(self, max_features: int) -> None:
        self.max_features = max_features
        self.vocabulary: dict[str, int] = {}
        self.idf = np.empty((0,), dtype=np.float32)

    def fit(self, texts: list[str]) -> None:
        documents = [set(text_terms(text)) for text in texts]
        frequencies = Counter(term for doc in documents for term in doc)
        # Deterministic tie-breaking, fit on training completions ONLY.
        terms = sorted(frequencies, key=lambda t: (-frequencies[t], t))[: self.max_features]
        self.vocabulary = {term: i for i, term in enumerate(terms)}
        self.idf = np.array(
            [math.log((1 + len(texts)) / (1 + frequencies[term])) + 1 for term in terms],
            dtype=np.float32,
        )
        if not terms:
            raise ValueError("No terms found in training text")

    def transform(self, texts: list[str]) -> np.ndarray:
        array = np.zeros((len(texts), len(self.vocabulary)), dtype=np.float32)
        for row, text in enumerate(texts):
            for term, count in Counter(text_terms(text)).items():
                if term in self.vocabulary:
                    array[row, self.vocabulary[term]] = count
        array *= self.idf
        length = np.linalg.norm(array, axis=1, keepdims=True)
        array /= np.maximum(length, 1e-12)
        return array


def grouped(rows: list[Candidate], split: str) -> dict[str, list[Candidate]]:
    result: dict[str, list[Candidate]] = defaultdict(list)
    for row in rows:
        if row.split == split:
            result[row.prompt_id].append(row)
    for items in result.values():
        items.sort(key=lambda candidate: candidate.candidate_index)
    return dict(result)


def pair_dataset(
    rows: dict[str, list[Candidate]],
    embeddings: dict[tuple[str, int], np.ndarray],
    *,
    pairs_per_prompt: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    pairs: list[np.ndarray] = []
    labels: list[float] = []
    usable_prompts = 0
    for _prompt_id, candidates in sorted(rows.items()):
        positive = [r for r in candidates if gold_binary(r.completion, r.target) == 1.0]
        negative = [r for r in candidates if gold_binary(r.completion, r.target) == 0.0]
        if not positive or not negative:
            continue
        usable_prompts += 1
        for _ in range(pairs_per_prompt):
            good = positive[int(rng.integers(len(positive)))]
            bad = negative[int(rng.integers(len(negative)))]
            choice = bool(rng.integers(2))
            ordered = (good, bad) if choice else (bad, good)
            pairs.append(np.stack([embeddings[(r.prompt_id, r.candidate_index)] for r in ordered]))
            labels.append(float(choice))
    if not pairs:
        raise ValueError(
            "No informative gold preference pairs: each split needs at least one "
            "prompt with both target-matching and nonmatching completions. "
            "Inspect candidates and choose a feasible binary gold task."
        )
    return np.stack(pairs).astype(np.float32), np.array(labels, dtype=np.float32), usable_prompts


class TextProxyReward(nnx.Module):
    def __init__(self, dimensions: int, rngs: nnx.Rngs) -> None:
        self.linear: nnx.Linear = nnx.Linear(dimensions, 1, use_bias=False, rngs=rngs)

    @override
    def __call__(self, features: Array) -> Array:
        return self.linear(features)[..., 0]


def best_of_n_text(
    test_rows: dict[str, list[Candidate]],
    embeddings: dict[tuple[str, int], np.ndarray],
    model: TextProxyReward,
) -> list[dict[str, float | int]]:
    # One fixed nested candidate pool per held-out prompt, independent of N.
    counts = {len(v) for v in test_rows.values()}
    if len(counts) != 1 or min(counts) < 2:
        raise ValueError("Each test prompt must have the same >=2 candidate count")
    count = counts.pop()
    available = [n for n in BEST_OF_N if n <= count]
    gold_rows, length_rows, proxy_rows = [], [], []
    for prompt_id, items in sorted(test_rows.items()):
        gold_rows.append([gold_binary(r.completion, r.target) for r in items])
        length_rows.append([len(r.token_ids) for r in items])
        features = np.stack([embeddings[(prompt_id, r.candidate_index)] for r in items])
        proxy_rows.append(np.asarray(model(jnp.asarray(features))))  # pyright: ignore[reportUnknownMemberType]
    gold = np.asarray(gold_rows, dtype=np.float32)
    length = np.asarray(length_rows, dtype=np.float32)
    proxy = np.asarray(proxy_rows, dtype=np.float32)
    indices = np.arange(len(gold))
    output: list[dict[str, float | int]] = []
    for n in available:
        chosen = np.argmax(proxy[:, :n], axis=1)
        longest = np.argmax(length[:, :n], axis=1)
        output.append(
            {
                "n": n,
                "sqrt_kl": float(np.sqrt(np.log(n) - (n - 1) / n)),
                "proxy_mean": float(np.mean(proxy[indices, chosen])),
                "gold_mean": float(np.mean(gold[indices, chosen])),
                "random_gold_mean": float(np.mean(gold[:, :n])),
                "oracle_gold_mean": float(np.mean(np.max(gold[:, :n], axis=1))),
                "length_only_gold_mean": float(np.mean(gold[indices, longest])),
                "selected_length_mean": float(np.mean(length[indices, chosen])),
            }
        )
    return output


def run_study(settings: Settings) -> tuple[list[dict[str, float | int]], dict[str, object]]:
    rows = load_cache(settings.cache_path)
    checkpoint_dir = settings.output_dir / "checkpoints"
    if checkpoint_dir.exists():
        raise FileExistsError(f"Run directory already contains checkpoints: {checkpoint_dir}")
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    texts = [f"{row.prompt}\n{row.completion}" for row in rows]
    training_texts = [text for text, row in zip(texts, rows, strict=True) if row.split == "train"]
    featurizer = TfidfFeatures(settings.max_features)
    featurizer.fit(training_texts)  # A test completion must NEVER enter fit.
    values = featurizer.transform(texts)
    embeddings = {
        (r.prompt_id, r.candidate_index): value for r, value in zip(rows, values, strict=True)
    }
    train_x, train_y, usable_train = pair_dataset(
        grouped(rows, "train"),
        embeddings,
        pairs_per_prompt=settings.train_pairs_per_prompt,
        seed=settings.seed,
    )
    valid_x, valid_y, usable_validation = pair_dataset(
        grouped(rows, "validation"),
        embeddings,
        pairs_per_prompt=settings.validation_pairs_per_prompt,
        seed=settings.seed + 1,
    )

    loader = PairBatches(
        train_x,
        train_y,
        settings.batch_size,
        settings.seed,
        epochs=settings.epochs,
        shuffle=True,
    )
    evaluation_loader = PairBatches(valid_x, valid_y, settings.batch_size, settings.seed)
    steps_per_epoch = (len(train_y) + settings.batch_size - 1) // settings.batch_size
    expected_steps = len(loader)

    def factory() -> tuple[TextProxyReward, nnx.Optimizer[TextProxyReward]]:
        new_model = TextProxyReward(len(featurizer.vocabulary), nnx.Rngs(settings.seed))
        optimizer = nnx.Optimizer(
            new_model,
            optax.adamw(settings.learning_rate, weight_decay=settings.weight_decay),
            wrt=nnx.Param,
        )
        return new_model, optimizer

    model, optimizer = factory()
    train_phase = make_train_phase(
        model=model,
        optimizer=optimizer,
        loader=loader,
        metrics=nnx.MultiMetric(loss=nnx.metrics.Average("loss")),
        loss_fn=preference_loss,
    )
    eval_phase = make_eval_phase(
        model=model,
        loader=evaluation_loader,
        metrics=nnx.MultiMetric(loss=nnx.metrics.Average("loss")),
        loss_fn=preference_loss,
    )
    history: list[dict[str, object]] = []
    with TrainerCheckpointer(checkpoint_dir) as checkpointer:
        hooks = (
            HookEntry(logging_hook(HistoryLogger("train", history)), every=steps_per_epoch),
            HookEntry(eval_hook(HistoryLogger("validation", history)), every=steps_per_epoch),
            checkpoint_entry(checkpointer, every=expected_steps),
        )
        run(make_runner(train=train_phase, eval=eval_phase, hooks=hooks))
        checkpointer.wait_until_finished()
        restored_model, restored_optimizer = factory()
        restored_step = restore_training_state(
            checkpointer, model=restored_model, optimizer=restored_optimizer
        )
        if restored_step != expected_steps:
            raise AssertionError("Incorrect restored optimizer step")
        model.eval()  # pyright: ignore[reportUnknownMemberType]
        restored_model.eval()  # pyright: ignore[reportUnknownMemberType]
        comparison = jnp.asarray(train_x[:2])  # pyright: ignore[reportUnknownMemberType]
        np.testing.assert_allclose(
            np.asarray(model(comparison)),
            np.asarray(restored_model(comparison)),
            rtol=1e-6,
            atol=1e-6,
        )

    valid_scores = np.asarray(model(jnp.asarray(valid_x)))  # pyright: ignore[reportUnknownMemberType]
    valid_predicted = valid_scores[:, 0] > valid_scores[:, 1]
    accuracy = float(np.mean(valid_predicted == (valid_y > 0)))
    bon = best_of_n_text(grouped(rows, "test"), embeddings, model)
    summary: dict[str, object] = {
        "settings": {
            **asdict(settings),
            "cache_path": str(settings.cache_path),
            "output_dir": str(settings.output_dir),
        },
        "source": "cached frozen GPT-2 text completions",
        "feature_type": "training-only TF-IDF unigram + bigram",
        "gold": "binary exact target phrase in completion (not human quality)",
        "held_out_preference_accuracy": accuracy,
        "usable_train_prompts": usable_train,
        "usable_validation_prompts": usable_validation,
        "test_prompt_count": len(grouped(rows, "test")),
        "restored_step": restored_step,
        "checkpoint_restoration_match": True,
        "caveat": "Methodology pilot; binary target matching and TF-IDF are not Gao et al. reward models.",
    }

    with (settings.output_dir / "best_of_n.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bon[0]))
        writer.writeheader()
        writer.writerows(bon)
    (settings.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Validated gold preference accuracy: {accuracy:.3f}")
    print(f"Saved BoN text study to {settings.output_dir}")
    return bon, summary


def main() -> None:
    run_study(tyro.cli(Settings))


if __name__ == "__main__":
    main()
