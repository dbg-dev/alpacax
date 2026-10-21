"""Inspect per-candidate gold and proxy scores from a cached-text Gao pilot.

Usage:
    python devtools/gao_rank_diagnostics.py \
        --run-dir runs/gao-weather/proxy-seed42
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def auc_positive_negative(rows: list[dict[str, object]]) -> float | None:
    """Probability a random gold-positive outranks a gold-negative.

    Ties receive half credit. Undefined for prompts with constant labels.
    """
    positives = [float(r["proxy_score"]) for r in rows if int(r["gold"]) == 1]
    negatives = [float(r["proxy_score"]) for r in rows if int(r["gold"]) == 0]
    if not positives or not negatives:
        return None
    correct = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in positives for n in negatives)
    return correct / (len(positives) * len(negatives))


def analyse(run_dir: Path) -> dict[str, object]:
    by_prompt: dict[str, list[dict[str, object]]] = defaultdict(list)
    with (run_dir / "held_out_candidates.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_prompt[row["prompt_id"]].append(
                {
                    "candidate_index": int(row["candidate_index"]),
                    "gold": int(row["gold"]),
                    "proxy_score": float(row["proxy_score"]),
                    "rank": int(row["rank"]),
                    "selected_n8": int(row["selected_n8"]),
                    "selected_n16": int(row["selected_n16"]),
                    "completion": row["completion"],
                }
            )

    prompts: list[dict[str, object]] = []
    mixed_auc: list[float] = []
    correct_pairs = 0.0
    all_pairs = 0
    for prompt_id, rows in sorted(by_prompt.items()):
        ranked = sorted(rows, key=lambda r: int(r["rank"]))
        positives = [r for r in ranked if int(r["gold"]) == 1]
        negatives = [r for r in ranked if int(r["gold"]) == 0]
        auc = auc_positive_negative(rows)
        if auc is not None:
            mixed_auc.append(auc)
            pairs = len(positives) * len(negatives)
            correct_pairs += auc * pairs
            all_pairs += pairs
        top = ranked[0]
        top_gold = positives[0] if positives else None
        n8 = next(r for r in rows if int(r["selected_n8"]) == 1)
        n16 = next(r for r in rows if int(r["selected_n16"]) == 1)
        prompts.append(
            {
                "prompt_id": prompt_id,
                "positive_count": len(positives),
                "candidate_count": len(rows),
                "auc": auc,
                "top_gold": int(top["gold"]),
                "n8_gold": int(n8["gold"]),
                "n16_gold": int(n16["gold"]),
                "best_positive_rank": int(top_gold["rank"]) if top_gold else None,
                "top_score": float(top["proxy_score"]),
                "best_positive_score": float(top_gold["proxy_score"]) if top_gold else None,
                "top_completion": str(top["completion"]),
                "best_positive_completion": str(top_gold["completion"]) if top_gold else None,
            }
        )

    weights = []
    with (run_dir / "feature_weights.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            weights.append({"feature": row["feature"], "weight": float(row["weight"])})

    rain = next((w for w in weights if w["feature"] == "rain"), None)
    return {
        "n_prompts": len(prompts),
        "prompts_with_both_labels": len(mixed_auc),
        "mean_within_prompt_auc": sum(mixed_auc) / len(mixed_auc) if mixed_auc else None,
        "pooled_within_prompt_auc": correct_pairs / all_pairs if all_pairs else None,
        "success_n8": sum(int(p["n8_gold"]) for p in prompts),
        "success_n16": sum(int(p["n16_gold"]) for p in prompts),
        "oracle_n16": sum(int(p["positive_count"] > 0) for p in prompts),
        "rain_feature": rain,
        "top_positive_weights": sorted(weights, key=lambda w: -w["weight"])[:12],
        "top_negative_weights": sorted(weights, key=lambda w: w["weight"])[:12],
        "prompts": prompts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyse(args.run_dir)
    (args.run_dir / "ranking_diagnostics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("PROMPT | positive/total | within-prompt AUC | best positive rank | N16 success")
    for row in report["prompts"]:
        auc = row["auc"]
        auc_str = f"{auc:.3f}" if isinstance(auc, float) else "n/a"
        print(
            f"{row['prompt_id']:<18} {row['positive_count']}/{row['candidate_count']}"
            f"   {auc_str:>5}  {str(row['best_positive_rank']):>5}"
            f"  {row['n16_gold']}"
        )
    print(
        "\nOVERALL:",
        json.dumps(
            {
                key: report[key]
                for key in (
                    "n_prompts",
                    "prompts_with_both_labels",
                    "mean_within_prompt_auc",
                    "pooled_within_prompt_auc",
                    "success_n8",
                    "success_n16",
                    "oracle_n16",
                    "rain_feature",
                )
            },
            ensure_ascii=False,
        ),
    )
    print("TOP POSITIVE WEIGHTS:", json.dumps(report["top_positive_weights"]))
    print("TOP NEGATIVE WEIGHTS:", json.dumps(report["top_negative_weights"]))
    print("\nSELECTED AND MISSED POSITIVE CONTINUATIONS:")
    for row in report["prompts"]:
        if row["positive_count"] == 0:
            continue
        print(f"--- {row['prompt_id']}, best positive rank={row['best_positive_rank']}")
        print("TOP SELECTED:", repr(row["top_completion"]))
        print("BEST GOLD-POSITIVE:", repr(row["best_positive_completion"]))
    print(f"\nSaved {args.run_dir / 'ranking_diagnostics.json'}")


if __name__ == "__main__":
    main()
