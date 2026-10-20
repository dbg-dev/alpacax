"""A tiny offline best-of-N reward-model experiment inspired by Gao et al. (2023).

This *synthetic sanity check* trains an independent Bradley-Terry proxy using
alpacax's NNX trainer, evaluates held-out preferences, saves/restores an Orbax
checkpoint and compares proxy-selected versus gold-selected candidates.

It is not a replication of Gao et al.'s pretrained language-model study:
synthetic two-feature candidates stand in for cached GPT-2 completions, and
an explicit nonlinear gold function stands in for a gold reward evaluator.
Run with: uv run --extra train python -m alpacax.examples.gao_proxy_sanity
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import override

import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax import nnx
from jax import Array

from alpacax.trainer.checkpoints import TrainerCheckpointer, restore_training_state
from alpacax.trainer.contracts import HookEntry
from alpacax.trainer.core import run
from alpacax.trainer.factories import make_eval_phase, make_runner, make_train_phase
from alpacax.trainer.hooks import checkpoint_entry, eval_hook, logging_hook

BEST_OF_N = (1, 2, 4, 8, 16, 32, 64, 128)
FEATURES = ("benefit", "length")


@dataclass(frozen=True, slots=True)
class Settings:
    seed: int = 42
    train_pairs: int = 2048
    validation_pairs: int = 256
    prompts: int = 128
    batch_size: int = 64
    epochs: int = 12
    learning_rate: float = 0.05
    output_dir: Path = Path("runs/gao-proxy-sanity")

    def __post_init__(self) -> None:
        if (
            min(self.train_pairs, self.validation_pairs, self.prompts, self.batch_size, self.epochs)
            <= 0
        ):
            raise ValueError("counts and epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


def gold_reward(candidates: np.ndarray) -> np.ndarray:
    """Synthetic gold: long answers help initially, then become harmful.

    Only training-length candidates (length < 0.8) are supplied as preferences;
    best-of-N candidates have a longer tail. This intentionally creates a
    blind spot for a linear proxy, *not* a claim about real language quality.
    """
    benefit, length = candidates[..., 0], candidates[..., 1]
    return benefit + 0.7 * length - 2.0 * np.maximum(length - 1.0, 0.0) ** 2


def preference_pairs(rng: np.random.Generator, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate independent gold-labelled preference pairs for the proxy."""
    benefit = rng.normal(0.0, 0.5, (count, 2))
    length = rng.uniform(0.0, 0.8, (count, 2))
    pairs = np.stack((benefit, length), axis=-1).astype(np.float32)
    labels = (gold_reward(pairs[:, 0]) > gold_reward(pairs[:, 1])).astype(np.float32)
    return pairs, labels


def candidate_pool(rng: np.random.Generator, prompts: int) -> np.ndarray:
    """A disjoint, held-out pool of 128 synthetic candidates per prompt."""
    shape = (prompts, BEST_OF_N[-1])
    benefit = rng.normal(0.0, 0.5, shape)
    length = np.abs(rng.normal(0.0, 0.8, shape))
    return np.stack((benefit, length), axis=-1).astype(np.float32)


@dataclass(frozen=True)
class PairBatches:
    pairs: np.ndarray
    labels: np.ndarray
    batch_size: int
    seed: int
    epochs: int = 1
    shuffle: bool = False

    def __len__(self) -> int:
        return self.epochs * ((len(self.labels) + self.batch_size - 1) // self.batch_size)

    def __iter__(self) -> Iterator[tuple[Array, Array]]:
        for epoch in range(self.epochs):
            indices = np.arange(len(self.labels))
            if self.shuffle:
                np.random.default_rng(self.seed + epoch).shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                yield jnp.asarray(self.pairs[batch]), jnp.asarray(self.labels[batch])  # pyright: ignore[reportUnknownMemberType]


class ProxyReward(nnx.Module):
    """Independent two-feature linear reward model; no shared generator weights."""

    def __init__(self, rngs: nnx.Rngs) -> None:
        self.linear: nnx.Linear = nnx.Linear(2, 1, use_bias=False, rngs=rngs)

    @override
    def __call__(self, candidates: Array) -> Array:
        return self.linear(candidates)[..., 0]


def preference_loss(model: nnx.Module, pairs: Array, labels: Array) -> tuple[Array, Array]:
    """Bradley-Terry negative log likelihood for pairwise gold preferences."""
    rewards = model(pairs)
    logits = rewards[:, 0] - rewards[:, 1]
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()  # pyright: ignore[reportUnknownMemberType]
    return loss, logits


class HistoryLogger:
    def __init__(self, name: str, rows: list[dict[str, object]]) -> None:
        self.name: str = name
        self.rows: list[dict[str, object]] = rows

    def log(self, metrics: dict[str, float], step: int) -> None:
        loss = metrics["loss"]
        self.rows.append({"step": step, "split": self.name, "loss": loss})
        print(f"step={step:4d} {self.name}_loss={loss:.4f}")

    def close(self) -> None:
        pass


def best_of_n_rows(pool: np.ndarray, proxy: np.ndarray) -> list[dict[str, float | int]]:
    """Compute paired BoN outcomes on the exact same held-out candidate pools."""
    gold = gold_reward(pool)
    index = np.arange(len(pool))
    output: list[dict[str, float | int]] = []
    for n in BEST_OF_N:
        predicted = proxy[:, :n]
        selected = np.argmax(predicted, axis=1)
        gold_oracle = np.max(gold[:, :n], axis=1)
        longest = np.argmax(pool[:, :n, 1], axis=1)
        # For with-replacement best-of-N, the paper uses the analytical KL.
        kl = np.log(n) - (n - 1) / n
        output.append(
            {
                "n": n,
                "sqrt_kl": float(np.sqrt(kl)),
                "proxy_mean": float(np.mean(predicted[index, selected])),
                "gold_mean": float(np.mean(gold[index, selected])),
                "random_gold_mean": float(np.mean(gold[:, :n])),
                "oracle_gold_mean": float(np.mean(gold_oracle)),  # pyright: ignore[reportUnknownArgumentType]
                "length_only_gold_mean": float(np.mean(gold[index, longest])),
                "selected_length_mean": float(np.mean(pool[index, selected, 1])),  # pyright: ignore[reportUnknownArgumentType]
            }
        )
    return output


def run_study(settings: Settings) -> tuple[list[dict[str, float | int]], dict[str, object]]:
    """Train/evaluate an independent proxy, verify Orbax restore, write results."""
    checkpoint_dir = settings.output_dir / "checkpoints"
    if checkpoint_dir.exists():
        raise FileExistsError(
            f"Checkpoint directory already exists: {checkpoint_dir}; "
            "choose a new --output-dir for a new run"
        )
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    # Independent RNG streams: no preference pair or candidate is reused.
    train_rng, validation_rng, candidate_rng = [
        np.random.default_rng(ss) for ss in np.random.SeedSequence(settings.seed).spawn(3)
    ]
    train_x, train_y = preference_pairs(train_rng, settings.train_pairs)
    valid_x, valid_y = preference_pairs(validation_rng, settings.validation_pairs)
    pool = candidate_pool(candidate_rng, settings.prompts)

    train_loader = PairBatches(
        train_x, train_y, settings.batch_size, settings.seed, epochs=settings.epochs, shuffle=True
    )
    validation_loader = PairBatches(valid_x, valid_y, settings.batch_size, settings.seed)
    steps_per_epoch = (settings.train_pairs + settings.batch_size - 1) // settings.batch_size
    expected_steps = len(train_loader)

    def make_model_and_optimizer() -> tuple[ProxyReward, nnx.Optimizer[ProxyReward]]:
        model = ProxyReward(nnx.Rngs(settings.seed))
        optimizer = nnx.Optimizer(model, optax.adam(settings.learning_rate), wrt=nnx.Param)
        return model, optimizer

    model, optimizer = make_model_and_optimizer()
    train = make_train_phase(
        model=model,
        optimizer=optimizer,
        loader=train_loader,
        metrics=nnx.MultiMetric(loss=nnx.metrics.Average("loss")),
        loss_fn=preference_loss,
    )
    validation = make_eval_phase(
        model=model,
        loader=validation_loader,
        metrics=nnx.MultiMetric(loss=nnx.metrics.Average("loss")),
        loss_fn=preference_loss,
    )
    history: list[dict[str, object]] = []

    with TrainerCheckpointer(checkpoint_dir) as checkpointer:
        hooks = (
            HookEntry(
                logging_hook(HistoryLogger("train", history)),
                every=steps_per_epoch,
                lead_only=False,
            ),
            HookEntry(
                eval_hook(HistoryLogger("validation", history)),
                every=steps_per_epoch,
                lead_only=False,
            ),
            checkpoint_entry(checkpointer, every=expected_steps),
        )
        run(make_runner(train=train, eval=validation, hooks=hooks))
        checkpointer.wait_until_finished()

        # A real NNX/Orbax restore test, not just a claim that a file was written.
        restored_model, restored_optimizer = make_model_and_optimizer()
        resumed_step = restore_training_state(
            checkpointer, model=restored_model, optimizer=restored_optimizer
        )
        if resumed_step != expected_steps:
            raise AssertionError(f"Expected restored step {expected_steps}, got {resumed_step}")

        model.eval()  # pyright: ignore[reportUnknownMemberType]
        restored_model.eval()  # pyright: ignore[reportUnknownMemberType]
        candidate_array = jnp.asarray(pool)  # pyright: ignore[reportUnknownMemberType]
        proxy = np.asarray(model(candidate_array))
        restored_proxy = np.asarray(restored_model(candidate_array))
        np.testing.assert_allclose(proxy, restored_proxy, rtol=1e-6, atol=1e-6)

    validation_rewards = np.asarray(model(jnp.asarray(valid_x)))  # pyright: ignore[reportUnknownMemberType]
    validation_logit = validation_rewards[:, 0] - validation_rewards[:, 1]
    preference_accuracy = float(np.mean((validation_logit > 0) == (valid_y > 0)))  # pyright: ignore[reportUnknownArgumentType]
    rows = best_of_n_rows(pool, proxy)
    summary: dict[str, object] = {
        "settings": {**asdict(settings), "output_dir": str(settings.output_dir)},
        "feature_names": FEATURES,
        "held_out_preference_accuracy": preference_accuracy,
        "restored_step": resumed_step,
        "checkpoint_restoration_match": True,
        "caveat": "Synthetic distribution-shift demonstration; no GPT-2 generations or scaling-law fit.",
    }

    with (settings.output_dir / "best_of_n.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (settings.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (settings.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"held_out_preference_accuracy={preference_accuracy:.3f}")
    print("N    proxy_mean   gold_mean   oracle_gold   selected_length")
    for row in rows:
        print(
            f"{row['n']:3d}   {row['proxy_mean']:10.3f}   {row['gold_mean']:9.3f}"
            f"   {row['oracle_gold_mean']:11.3f}   {row['selected_length_mean']:15.3f}"
        )
    print(f"Saved results and Orbax checkpoint to {settings.output_dir}")
    return rows, summary


def main() -> None:
    run_study(tyro.cli(Settings))


if __name__ == "__main__":
    main()
