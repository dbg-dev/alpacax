"""End-to-end tests for the deliberately synthetic Gao-inspired training demo."""

from pathlib import Path

import numpy as np
import pytest

from alpacax.examples.gao_proxy_sanity import (
    BEST_OF_N,
    Settings,
    best_of_n_rows,
    gold_reward,
    preference_pairs,
    run_study,
)


def test_gold_reward_rewards_moderate_length_but_penalises_extremes() -> None:
    features = np.array([[0.0, 0.8], [0.0, 2.0]], dtype=np.float32)
    scores = gold_reward(features)
    assert scores[0] > scores[1]


def test_pair_labels_use_gold_evaluator() -> None:
    pairs, labels = preference_pairs(np.random.default_rng(7), 32)
    assert pairs.shape == (32, 2, 2)
    assert labels.shape == (32,)
    np.testing.assert_array_equal(
        labels,
        gold_reward(pairs[:, 0]) > gold_reward(pairs[:, 1]),
    )
    assert np.all(pairs[:, :, 1] < 0.8)


def test_best_of_n_controls_share_candidate_pool() -> None:
    rng = np.random.default_rng(19)
    pool = np.stack(
        [rng.normal(0.0, 0.5, (16, 128)), np.abs(rng.normal(0.0, 0.8, (16, 128)))],
        axis=-1,
    ).astype(np.float32)
    proxy = pool[..., 0] + 0.7 * pool[..., 1]
    rows = best_of_n_rows(pool, proxy)
    assert [row["n"] for row in rows] == list(BEST_OF_N)
    assert rows[0]["sqrt_kl"] == 0.0
    assert all(
        row["oracle_gold_mean"] >= row["gold_mean"] - 1e-5 for row in rows
    )
    assert rows[-1]["proxy_mean"] > rows[0]["proxy_mean"]


def test_end_to_end_training_checkpoint_and_results(tmp_path: Path) -> None:
    settings = Settings(
        seed=11,
        train_pairs=128,
        validation_pairs=64,
        prompts=12,
        batch_size=32,
        epochs=4,
        output_dir=tmp_path / "gao-sanity",
    )
    rows, summary = run_study(settings)
    assert len(rows) == len(BEST_OF_N)
    assert summary["restored_step"] == 16
    assert summary["checkpoint_restoration_match"] is True
    accuracy = summary["held_out_preference_accuracy"]
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0
    assert rows[-1]["proxy_mean"] > rows[0]["proxy_mean"]

    for name in ("best_of_n.csv", "summary.json", "history.json"):
        assert (settings.output_dir / name).is_file()
    assert (settings.output_dir / "checkpoints").is_dir()

    # Never silently overwrite checkpoints when reusing a results directory.
    with pytest.raises(FileExistsError, match="choose a new"):
        run_study(settings)
