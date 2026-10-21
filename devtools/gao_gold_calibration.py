"""Diagnose gold-label diversity in cached GPT-2 continuations.

Uses the same case-insensitive, word-bounded literal target test as
gao_text_best_of_n.gold_binary. No model fitting and no API calls.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def gold_hit(completion: str, target: str) -> int:
    pattern = r"(?<!\w)" + re.escape(target.strip()) + r"(?!\w)"
    return int(bool(re.search(pattern, completion, flags=re.IGNORECASE)))


def prompt_statistics(labels: list[int]) -> dict[str, float | int | bool]:
    """For binary gold, within-prompt variance is p(1-p)."""
    if not labels or any(value not in (0, 1) for value in labels):
        raise ValueError("Expected a nonempty list of binary gold labels")
    n = len(labels)
    positive = sum(labels)
    p = positive / n
    return {
        "candidates": n,
        "positive": positive,
        "negative": n - positive,
        "positive_rate": p,
        "gold_variance": p * (1 - p),
        "oracle_gold": max(labels),
        "random_gold": p,
        "oracle_headroom": max(labels) - p,
        "mixed": 0 < positive < n,
    }


def inspect_cache(cache_path: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    prompt_targets: dict[str, tuple[str, str]] = {}
    with cache_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            row: dict[str, Any] = json.loads(raw_line)
            for field in ("prompt_id", "split", "target", "completion"):
                if field not in row:
                    raise ValueError(f"Line {line_number}: missing {field}")
            prompt_id = str(row["prompt_id"])
            split = str(row["split"])
            target = str(row["target"])
            if split not in ("train", "validation", "test") or not target.strip():
                raise ValueError(f"Line {line_number}: invalid split or empty target")
            identity = (split, target)
            if prompt_id in prompt_targets and prompt_targets[prompt_id] != identity:
                raise ValueError(f"Line {line_number}: inconsistent prompt metadata")
            prompt_targets[prompt_id] = identity
            grouped[(split, prompt_id)].append(gold_hit(str(row["completion"]), target))

    if not grouped:
        raise ValueError(f"Empty candidate cache: {cache_path}")

    prompts = [
        {"split": split, "prompt_id": prompt_id, **prompt_statistics(labels)}
        for (split, prompt_id), labels in sorted(grouped.items())
    ]

    by_split: dict[str, dict[str, float | int]] = {}
    for split in ("train", "validation", "test"):
        rows = [r for r in prompts if r["split"] == split]
        if not rows:
            continue
        count = len(rows)
        by_split[split] = {
            "prompts": count,
            "mixed_prompts": sum(int(r["mixed"]) for r in rows),
            "mean_positive_rate": sum(float(r["positive_rate"]) for r in rows) / count,
            "mean_within_prompt_variance": sum(float(r["gold_variance"]) for r in rows) / count,
            "mean_oracle_headroom": sum(float(r["oracle_headroom"]) for r in rows) / count,
        }

    return {"cache_path": str(cache_path), "by_split": by_split, "prompts": prompts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("runs/gao-weather/candidates.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("runs/gao-weather/calibration.json"),
    )
    args = parser.parse_args()
    result = inspect_cache(args.cache_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("Split        Prompts  Mixed  Mean p  Mean variance  Oracle headroom")
    for split, stats in result["by_split"].items():
        print(
            f"{split:<13}{stats['prompts']:<9}{stats['mixed_prompts']:<7}"
            f"{stats['mean_positive_rate']:<8.3f}"
            f"{stats['mean_within_prompt_variance']:<15.3f}"
            f"{stats['mean_oracle_headroom']:.3f}"
        )
    print("\nPer-prompt results:")
    for row in result["prompts"]:
        print(
            f"{row['split']:>10} {row['prompt_id']:<23}"
            f" {row['positive']}/{row['candidates']} positive"
            f"  variance={row['gold_variance']:.3f}"
            f"  headroom={row['oracle_headroom']:.3f}"
        )
    print(f"\nSaved {args.output_path}")


if __name__ == "__main__":
    main()
