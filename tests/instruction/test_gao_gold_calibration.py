"""Check the standalone Gao calibration CLI against a small synthetic cache."""

import json
import subprocess
import sys
from pathlib import Path


def test_calibration_reports_gold_diversity(tmp_path: Path) -> None:
    entries = [
        {"prompt_id": "p1", "split": "train", "target": "rain", "completion": "RAIN"},
        {"prompt_id": "p1", "split": "train", "target": "rain", "completion": "sun"},
        {"prompt_id": "p2", "split": "validation", "target": "rain", "completion": "rainy"},
        {"prompt_id": "p2", "split": "validation", "target": "rain", "completion": "drain"},
        {"prompt_id": "p3", "split": "test", "target": "rain", "completion": "rain"},
        {"prompt_id": "p3", "split": "test", "target": "rain", "completion": "rain."},
    ]
    cache = tmp_path / "candidates.jsonl"
    output = tmp_path / "summary.json"
    cache.write_text("\n".join(json.dumps(r) for r in entries) + "\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[2] / "devtools" / "gao_gold_calibration.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--cache-path",
            str(cache),
            "--output-path",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["by_split"]["train"]["mixed_prompts"] == 1
    assert result["by_split"]["train"]["mean_positive_rate"] == 0.5
    assert result["by_split"]["train"]["mean_within_prompt_variance"] == 0.25
    assert result["by_split"]["train"]["mean_oracle_headroom"] == 0.5
    assert result["by_split"]["validation"]["mixed_prompts"] == 0
    assert result["by_split"]["validation"]["mean_positive_rate"] == 0
    assert result["by_split"]["test"]["mixed_prompts"] == 0
    assert result["by_split"]["test"]["mean_positive_rate"] == 1
    assert result["by_split"]["test"]["mean_oracle_headroom"] == 0
