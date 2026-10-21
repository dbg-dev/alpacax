"""Tests for the standalone Gao gold-reward calibration report."""

import json

import pytest

from devtools.gao_gold_calibration import gold_hit, inspect_cache, prompt_statistics


def test_gold_hit_is_bounded_and_case_insensitive():
    assert gold_hit("There will be RAIN today.", "rain") == 1
    assert gold_hit("It might be rainy tomorrow.", "rain") == 0
    assert gold_hit("The water will drain away.", "rain") == 0


@pytest.mark.parametrize(
    ("labels", "rate", "variance", "headroom"),
    [
        ([0, 0, 0, 0], 0, 0, 0),
        ([1, 1, 1, 1], 1, 0, 0),
        ([0, 1, 0, 1], 0.5, 0.25, 0.5),
    ],
)
def test_prompt_statistics(labels, rate, variance, headroom):
    metrics = prompt_statistics(labels)
    assert metrics["positive_rate"] == rate
    assert metrics["gold_variance"] == variance
    assert metrics["oracle_headroom"] == headroom


def test_inspect_cache_reports_within_prompt_diversity(tmp_path):
    entries = [
        {"prompt_id": "p1", "split": "train", "target": "rain", "completion": "rain"},
        {"prompt_id": "p1", "split": "train", "target": "rain", "completion": "sun"},
        {"prompt_id": "p2", "split": "test", "target": "rain", "completion": "rain"},
        {"prompt_id": "p2", "split": "test", "target": "rain", "completion": "rain"},
    ]
    path = tmp_path / "cache.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in entries) + "\n", encoding="utf-8")
    result = inspect_cache(path)
    assert result["by_split"]["train"]["mixed_prompts"] == 1
    assert result["by_split"]["train"]["mean_oracle_headroom"] == 0.5
    assert result["by_split"]["test"]["mixed_prompts"] == 0
    assert result["by_split"]["test"]["mean_oracle_headroom"] == 0
