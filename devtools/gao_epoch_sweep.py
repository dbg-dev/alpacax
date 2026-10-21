"""Summarise a fixed-seed Bradley-Terry optimisation-duration sweep.

For each feature source, runs at 10, 50 and 100 epochs use the same candidate
cache, preference sampling, model initialization, optimiser and loss. Report
test results descriptively, never select checkpoints by test score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import TypedDict

FEATURE_SOURCES = ("prompt_and_completion", "completion_only")
EPOCHS = (10, 50, 100)


class SweepRow(TypedDict):
    feature_source: str
    epochs: int
    validation_accuracy: float
    train_loss: float
    validation_loss: float
    min_validation_loss: float
    epoch_of_min_validation_loss: int
    mean_test_auc: float
    gold_at_8: float
    gold_at_16: float
    oracle_at_16: float
    rain_weight: float
    rain_weight_fraction: float
    train_pairs_per_prompt: int
    validation_pairs_per_prompt: int


def read_run(root: Path, source: str, epochs: int) -> SweepRow:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    ranks = json.loads((root / "ranking_diagnostics.json").read_text(encoding="utf-8"))
    history = json.loads((root / "history.json").read_text(encoding="utf-8"))
    with (root / "best_of_n.csv").open(newline="", encoding="utf-8") as handle:
        bon = {int(row["n"]): row for row in csv.DictReader(handle)}
    with (root / "feature_weights.csv").open(newline="", encoding="utf-8") as handle:
        weights = {row["feature"]: float(row["weight"]) for row in csv.DictReader(handle)}
    settings = summary["settings"]
    if settings["epochs"] != epochs or settings["feature_source"] != source:
        raise ValueError(f"Unexpected settings for {root}")
    if not summary["checkpoint_restoration_match"]:
        raise ValueError(f"Checkpoint failed validation: {root}")
    for n in (8, 16):
        if n not in bon:
            raise ValueError(f"Missing best-of-N={n} for {root}")
    records = {
        split: [float(x["loss"]) for x in history if x["split"] == split]
        for split in ("train", "validation")
    }
    if any(len(v) != epochs for v in records.values()):
        raise ValueError(f"Expected {epochs} train and validation loss records for {root}")
    validation = records["validation"]
    best_index = min(range(len(validation)), key=validation.__getitem__)
    norm = math.sqrt(sum(value * value for value in weights.values()))
    rain = weights.get("rain", 0.0)
    auc = ranks["mean_within_prompt_auc"]
    if auc is None:
        raise ValueError(f"No informative held-out prompts for {root}")
    return {
        "feature_source": source,
        "epochs": epochs,
        "validation_accuracy": float(summary["held_out_preference_accuracy"]),
        "train_loss": records["train"][-1],
        "validation_loss": validation[-1],
        "min_validation_loss": validation[best_index],
        "epoch_of_min_validation_loss": best_index + 1,
        "mean_test_auc": float(auc),
        "gold_at_8": float(bon[8]["gold_mean"]),
        "gold_at_16": float(bon[16]["gold_mean"]),
        "oracle_at_16": float(bon[16]["oracle_gold_mean"]),
        "rain_weight": rain,
        "rain_weight_fraction": rain / norm if norm else 0.0,
        "train_pairs_per_prompt": int(settings["train_pairs_per_prompt"]),
        "validation_pairs_per_prompt": int(settings["validation_pairs_per_prompt"]),
    }


def analyse(base: Path) -> list[SweepRow]:
    rows = [
        read_run(base / f"{source}-ep{epochs}", source, epochs)
        for source in FEATURE_SOURCES
        for epochs in EPOCHS
    ]
    if len({r["oracle_at_16"] for r in rows}) != 1:
        raise ValueError("Gold oracle differs across supposedly identical candidate pools")
    if any(r["train_pairs_per_prompt"] != rows[0]["train_pairs_per_prompt"] for r in rows):
        raise ValueError("Training-pair budgets differ")
    if any(r["validation_pairs_per_prompt"] != rows[0]["validation_pairs_per_prompt"] for r in rows):
        raise ValueError("Validation-pair budgets differ")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = analyse(args.base_dir)
    with (args.base_dir / "epoch_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SweepRow.__annotations__))
        writer.writeheader()
        writer.writerows(rows)
    chosen = {
        source: min(
            (r for r in rows if r["feature_source"] == source),
            key=lambda r: r["validation_loss"],
        )["epochs"]
        for source in FEATURE_SOURCES
    }
    (args.base_dir / "epoch_sweep.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "selected_epochs_by_final_validation_loss": chosen,
                "caveat": (
                    "Exploratory test evaluation: these held-out prompts have been inspected "
                    "repeatedly. Reconfirm model selection on fresh, untouched prompts."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("FEATURES | EPOCHS | TRAIN LOSS | VALID LOSS | VALID ACC | TEST AUC | GOLD@16 | RAIN/NORM")
    for r in rows:
        print(
            f"{r['feature_source']:<22} {r['epochs']:>3} "
            f"{r['train_loss']:.4f} {r['validation_loss']:.4f} "
            f"{r['validation_accuracy']:.3f} {r['mean_test_auc']:.3f} "
            f"{r['gold_at_16']:.3f} {r['rain_weight_fraction']:.3f}"
        )
    print("Validation-selected epoch count (not selected on test):", chosen)
    print("Saved epoch_sweep.csv and epoch_sweep.json")


if __name__ == "__main__":
    main()
