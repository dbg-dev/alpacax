"""Offline tests for frozen GPT-2 cache and text-proxy research pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from flax import nnx

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
    model.eval()
    args = dict(
        prompt_ids=[1, 2, 3], max_new_tokens=4, top_k=8, temperature=0.9,
        seed=123, eos_token_id=31,
    )
    together = generate_token_ids(model, start_index=0, count=2, **args)
    separate = [
        *generate_token_ids(model, start_index=0, count=1, **args),
        *generate_token_ids(model, start_index=1, count=1, **args),
    ]
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


def _cache(tmp_path: Path) -> Path:
    cache = tmp_path / "candidates.jsonl"
    content = []
    for split in ("train", "validation", "test"):
        for candidate_id, completion in enumerate(
            ["and the sailor left", "a sailor left", "the sailor arrived", "quietly"]
        ):
            content.append(
                {
                    "prompt_id": f"{split}-p0", "split": split,
                    "prompt": "At the harbor: ", "target": "the",
                    "candidate_index": candidate_id, "completion": completion,
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
        cache_path=cache, output_dir=tmp_path / "results",
        train_pairs_per_prompt=8, validation_pairs_per_prompt=4,
        epochs=2, batch_size=4, max_features=16,
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
    records = [json.loads(row) for row in cache.read_text().splitlines()]
    for row in records:
        if row["split"] == "validation":
            row["completion"] = "no target word"
    cache.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No informative gold preference pairs"):
        run_study(
            StudySettings(cache_path=cache, output_dir=tmp_path / "results", epochs=1)
        )
