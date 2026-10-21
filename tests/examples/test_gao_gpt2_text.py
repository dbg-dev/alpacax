"""Offline tests for frozen GPT-2 cache and text-proxy research pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from flax import nnx

from alpacax.examples._prompt_isolation import check_prompt_isolation
from alpacax.examples.gao_gpt2_candidates import (
    Settings as CacheSettings,
)
from alpacax.examples.gao_gpt2_candidates import (
    generate_token_ids,
    load_prompts,
    prompt_seed,
)
from alpacax.examples.gao_text_best_of_n import (
    Settings as StudySettings,
)
from alpacax.examples.gao_text_best_of_n import (
    TfidfFeatures,
    gold_binary,
    load_cache,
    run_study,
)
from alpacax.models.gpt2.config import (
    AttentionConfig,
    DenseMLPConfig,
    EmbeddingConfig,
    GPTConfig,
    TransformerBlockConfig,
)
from alpacax.models.gpt2.gpt2 import GPT2


def test_prompt_ids_are_unique_across_splits(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    values = [
        {"prompt_id": "shared", "split": split, "prompt": "Hello", "target": "the"}
        for split in ("train", "validation", "test")
    ]
    path.write_text("\n".join(json.dumps(v) for v in values), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate prompt_id"):
        load_prompts(path)


def test_generation_keys_stable_and_cache_settings_valid() -> None:
    assert prompt_seed(42, "a") == prompt_seed(42, "a")
    assert prompt_seed(42, "a") != prompt_seed(42, "b")
    with pytest.raises(ValueError):
        CacheSettings(top_k=0)


def test_tiny_nnx_sampler_repeatable_across_batch_sizes() -> None:
    config = GPTConfig(
        embedding=EmbeddingConfig(vocab_size=32, d_model=8, dropout_rate=0.0),
        block=TransformerBlockConfig(
            attention=AttentionConfig(d_model=8, n_heads=2, dropout_rate=0.0),
            mlp=DenseMLPConfig(d_model=8, expansion_factor=2.0),
            dropout_rate=0.0,
        ),
        n_layers=1,
        max_seq_len=16,
    )
    model = GPT2(config, nnx.Rngs(0))
    model.eval()  # pyright: ignore[reportUnknownMemberType]

    def sample(start_index: int, count: int) -> list[list[int]]:
        return generate_token_ids(
            model,
            [1, 2, 3],
            start_index=start_index,
            count=count,
            max_new_tokens=4,
            top_k=8,
            temperature=0.9,
            seed=123,
            eos_token_id=31,
        )

    together = sample(0, 2)
    separate = [*sample(0, 1), *sample(1, 1)]
    assert together == separate
    assert all(len(row) <= 4 for row in together)


def test_text_gold_is_exact_phrase_not_substring() -> None:
    assert gold_binary("The sea and the mountains", "the") == 1
    assert gold_binary("something else", "the") == 0
    assert gold_binary("and", "and") == 1


def test_tfidf_uses_train_only() -> None:
    encoder = TfidfFeatures(8)
    encoder.fit(["the tea was hot", "hot tea in the cup"])
    before = dict(encoder.vocabulary)
    vectors = encoder.transform(["unknown secretlyheldoutword"])
    assert encoder.vocabulary == before
    assert vectors.shape == (1, len(before))
    assert np.all(vectors == 0)


_SPLIT_PROMPTS = {
    "train": "At the harbor: ",
    "validation": "By the quay: ",
    "test": "Near the pier: ",
}


def _cache(tmp_path: Path) -> Path:
    cache = tmp_path / "candidates.jsonl"
    content: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        for candidate_id, completion in enumerate(
            ["and the sailor left", "a sailor left", "the sailor arrived", "quietly"]
        ):
            content.append(
                {
                    "prompt_id": f"{split}-p0",
                    "split": split,
                    "prompt": _SPLIT_PROMPTS[split],
                    "target": "the",
                    "candidate_index": candidate_id,
                    "completion": completion,
                    "token_ids": [10, 20, 30][: (candidate_id % 3) + 1],
                    "model_name": "tiny-test",
                }
            )
    cache.write_text("\n".join(json.dumps(row) for row in content) + "\n", encoding="utf-8")
    return cache


def test_text_study_end_to_end_with_real_nnx_train_and_restore(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    rows = load_cache(cache)
    assert len(rows) == 12
    settings = StudySettings(
        cache_path=cache,
        output_dir=tmp_path / "results",
        train_pairs_per_prompt=8,
        validation_pairs_per_prompt=4,
        epochs=2,
        batch_size=4,
        max_features=16,
    )
    bon, summary = run_study(settings)
    assert [row["n"] for row in bon] == [1, 2, 4]
    assert summary["restored_step"] == 4
    assert summary["feature_type"] == "training-only TF-IDF unigram + bigram"
    assert summary["usable_train_prompts"] == 1
    assert (settings.output_dir / "best_of_n.csv").is_file()
    assert (settings.output_dir / "history.json").is_file()
    assert (settings.output_dir / "summary.json").is_file()
    with pytest.raises(FileExistsError):
        run_study(settings)


def test_missing_gold_preference_pairs_fail_loudly(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    records = [cast(dict[str, object], json.loads(row)) for row in cache.read_text().splitlines()]
    for row in records:
        if row["split"] == "validation":
            row["completion"] = "no target word"
    cache.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No informative gold preference pairs"):
        run_study(StudySettings(cache_path=cache, output_dir=tmp_path / "results", epochs=1))


def _write_prompts(path: Path, rows: list[tuple[str, str, str]]) -> None:
    values = [
        {"prompt_id": prompt_id, "split": split, "prompt": text, "target": "the"}
        for prompt_id, split, text in rows
    ]
    path.write_text("\n".join(json.dumps(v) for v in values), encoding="utf-8")


def test_load_prompts_rejects_same_text_in_different_splits(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    _write_prompts(
        path,
        [
            ("train-1", "train", "At the harbor: "),
            ("validation-1", "validation", "By the quay: "),
            # Different ID, but the same text as train-1 apart from case and spacing.
            ("test-1", "test", "  at the   HARBOR:"),
        ],
    )
    with pytest.raises(ValueError, match=r"'train-1' \(train\) and 'test-1' \(test\)"):
        load_prompts(path)


def test_load_prompts_allows_repeated_text_within_one_split(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    _write_prompts(
        path,
        [
            ("train-1", "train", "At the harbor: "),
            ("train-2", "train", "At the harbor: "),
            ("validation-1", "validation", "By the quay: "),
            ("test-1", "test", "Near the pier: "),
        ],
    )
    assert len(load_prompts(path)) == 4


def test_shipped_example_prompts_still_load() -> None:
    example = Path(__file__).parents[2] / "docs" / "gao-prompts.example.jsonl"
    assert len(load_prompts(example)) > 0


def test_load_cache_rejects_same_prompt_text_in_different_splits(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    records = [cast(dict[str, object], json.loads(row)) for row in cache.read_text().splitlines()]
    for row in records:
        row["prompt"] = _SPLIT_PROMPTS["train"]  # test/validation now reuse train's text
    cache.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Prompt text appears in more than one split"):
        load_cache(cache)


def test_load_cache_missing_split_message_is_accurate(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    records = [cast(dict[str, object], json.loads(row)) for row in cache.read_text().splitlines()]
    kept = [row for row in records if row["split"] != "test"]
    cache.write_text("\n".join(json.dumps(row) for row in kept) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain train, validation and test"):
        load_cache(cache)


def test_prompt_isolation_reports_first_clashes_and_counts_the_rest() -> None:
    entries = [(f"train-{i}", "train", f"prompt {i}") for i in range(8)]
    entries += [(f"test-{i}", "test", f"prompt {i}") for i in range(8)]
    with pytest.raises(ValueError, match=r"\(\+3 more\)") as info:
        check_prompt_isolation(entries)
    assert "'train-0' (train) and 'test-0' (test)" in str(info.value)
