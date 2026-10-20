"""Cache completions from frozen, strictly mapped GPT-2 for a Gao-style BoN study.

From repository root (downloads GPT-2 124M weights/tokenizer once):
  uv run --extra train --extra huggingface --with 'transformers>=4.57,<5' \
      python -m alpacax.examples.gao_gpt2_candidates \
      --prompts-path docs/gao-prompts.example.jsonl \
      --output-path runs/gao-gpt2/candidates.jsonl

This stage *replaces synthetic candidate generation*, not the independent
proxy reward model or the gold evaluator. The follow-up text study consumes
the cached completions. No weights of the generator are trained.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from flax import nnx
from jax import Array

from alpacax.integrations.huggingface import model_mapper_or_raise
from alpacax.models.gpt2.config import (
    AttentionConfig,
    DenseMLPConfig,
    EmbeddingConfig,
    GPTConfig,
    TransformerBlockConfig,
)
from alpacax.models.gpt2.gpt2 import GPT2
from alpacax.models.gpt2.hf_mappings import hfgpt2_to_local_gpt2_rules

SPLITS = frozenset(("train", "validation", "test"))


@dataclass(frozen=True, slots=True)
class Prompt:
    prompt_id: str
    split: str
    prompt: str
    target: str


@dataclass(frozen=True, slots=True)
class Settings:
    prompts_path: Path = Path("docs/gao-prompts.example.jsonl")
    output_path: Path = Path("runs/gao-gpt2/candidates.jsonl")
    model_name: str = "openai-community/gpt2"
    seed: int = 42
    candidates_per_prompt: int = 8
    max_new_tokens: int = 20
    top_k: int = 40
    temperature: float = 1.0
    batch_size: int = 4

    def __post_init__(self) -> None:
        if min(self.candidates_per_prompt, self.max_new_tokens, self.batch_size, self.top_k) < 1:
            raise ValueError("candidates, tokens, top_k and batch_size must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


def load_prompts(path: Path) -> list[Prompt]:
    prompts: list[Prompt] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            item = json.loads(raw_line)
            prompt = Prompt(**item)
            if not all((prompt.prompt_id.strip(), prompt.prompt.strip(), prompt.target.strip())):
                raise ValueError(f"Line {number}: prompt_id, prompt and target must be nonempty")
            if prompt.split not in SPLITS:
                raise ValueError(f"Line {number}: unsupported split {prompt.split!r}")
            if prompt.prompt_id in seen:
                raise ValueError(f"Line {number}: duplicate prompt_id {prompt.prompt_id!r}")
            seen.add(prompt.prompt_id)
            prompts.append(prompt)
    if not prompts:
        raise ValueError("Prompt file is empty")
    if {p.split for p in prompts} != SPLITS:
        raise ValueError("Provide separate train, validation and test prompt_ids")
    return prompts


def prompt_seed(base_seed: int, prompt_id: str) -> int:
    """Stable across runs and independent of input file ordering."""
    hash_prefix = hashlib.sha256(prompt_id.encode("utf-8")).digest()[:4]
    return (base_seed ^ int.from_bytes(hash_prefix, "big")) & 0xFFFFFFFF


def gpt2_from_huggingface(model_name: str) -> tuple[GPT2, Any]:
    """Strictly load all pretrained GPT-2 weights into alpacax's NNX model."""
    transformers = importlib.import_module("transformers")
    hf_model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained(model_name)
    hf_config = hf_model.config
    if hf_config.model_type != "gpt2":
        raise ValueError("Only GPT-2 architecture is supported by these weight rules")
    dim = int(hf_config.n_embd)
    model_config = GPTConfig(
        embedding=EmbeddingConfig(
            vocab_size=int(hf_config.vocab_size), d_model=dim, dropout_rate=0.0
        ),
        block=TransformerBlockConfig(
            attention=AttentionConfig(d_model=dim, n_heads=int(hf_config.n_head), dropout_rate=0.0),
            mlp=DenseMLPConfig(d_model=dim, expansion_factor=4.0),
            dropout_rate=0.0,
        ),
        n_layers=int(hf_config.n_layer),
        max_seq_len=int(hf_config.n_positions),
        tie_embeddings=True,
    )
    local = GPT2(model_config, nnx.Rngs(0))
    model = model_mapper_or_raise(
        hf_model,
        local,
        hfgpt2_to_local_gpt2_rules(int(hf_config.n_layer)),
        strict=True,
        allow_unused_src=("lm_head.weight",),
    )
    if not isinstance(model, GPT2):
        raise TypeError("Unexpected mapped model type")
    model.eval()  # pyright: ignore[reportUnknownMemberType]
    return model, tokenizer


@nnx.jit
def next_token_logits(model: GPT2, padded_ids: Array, position: Array) -> Array:
    """Full-context forward, fixed compiled shape; reads only current last logits.

    This is intentionally simple rather than a KV-cache sampler. Right-side
    future pad tokens do not influence earlier positions in causal GPT-2.
    """
    all_logits = model(padded_ids)
    return jax.lax.dynamic_index_in_dim(all_logits, position, axis=1, keepdims=False)


def generate_token_ids(
    model: GPT2,
    prompt_ids: list[int],
    *,
    start_index: int,
    count: int,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    seed: int,
    eos_token_id: int,
) -> list[list[int]]:
    """Sample a batch; per-candidate keys make outputs invariant to batch_size."""
    if not prompt_ids:
        raise ValueError("Empty tokenized prompt")
    if len(prompt_ids) + max_new_tokens > model.max_seq_len:
        raise ValueError("Prompt plus new tokens exceeds GPT-2 context window")
    if top_k > int(model.token_embedding.num_embeddings):
        raise ValueError("top_k exceeds vocabulary size")

    length = len(prompt_ids) + max_new_tokens
    padded = np.full((count, length), eos_token_id, dtype=np.int32)
    padded[:, : len(prompt_ids)] = np.array(prompt_ids, dtype=np.int32)
    tokens = jnp.asarray(padded)
    master = jax.random.PRNGKey(seed)
    keys = jnp.stack([jax.random.fold_in(master, start_index + i) for i in range(count)])

    for position in range(len(prompt_ids), length):
        logits = next_token_logits(model, tokens, jnp.asarray(position - 1, dtype=jnp.int32))
        best_logits, best_ids = jax.lax.top_k(logits, top_k)
        split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(keys)
        keys, sample_keys = split_keys[:, 0], split_keys[:, 1]
        relative_ids = jax.vmap(jax.random.categorical)(sample_keys, best_logits / temperature)
        sampled_ids = jnp.take_along_axis(best_ids, relative_ids[:, None], axis=1)[:, 0]
        tokens = tokens.at[:, position].set(sampled_ids)

    all_tokens = np.asarray(tokens)
    continuations: list[list[int]] = []
    for tokens_row in all_tokens[:, len(prompt_ids) :].tolist():
        # Include only generated text, stop at EOS; pad tokens never enter result.
        if eos_token_id in tokens_row:
            tokens_row = tokens_row[: tokens_row.index(eos_token_id)]
        continuations.append([int(t) for t in tokens_row])
    return continuations


def build_cache(settings: Settings) -> int:
    prompts = load_prompts(settings.prompts_path)
    if settings.output_path.exists():
        raise FileExistsError(f"Candidate cache already exists: {settings.output_path}")
    model, tokenizer = gpt2_from_huggingface(settings.model_name)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id")
    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = settings.output_path.with_suffix(".partial")
    if tmp_path.exists():
        raise FileExistsError(f"Remove incomplete previous cache first: {tmp_path}")

    written = 0
    try:
        with tmp_path.open("x", encoding="utf-8") as target:
            for prompt in prompts:
                prefix = list(tokenizer.encode(prompt.prompt, add_special_tokens=False))
                if len(prefix) + settings.max_new_tokens > model.max_seq_len:
                    raise ValueError(f"Prompt {prompt.prompt_id!r} exceeds context budget")
                seed = prompt_seed(settings.seed, prompt.prompt_id)
                for first in range(0, settings.candidates_per_prompt, settings.batch_size):
                    count = min(settings.batch_size, settings.candidates_per_prompt - first)
                    generated = generate_token_ids(
                        model,
                        prefix,
                        start_index=first,
                        count=count,
                        max_new_tokens=settings.max_new_tokens,
                        top_k=settings.top_k,
                        temperature=settings.temperature,
                        seed=seed,
                        eos_token_id=int(tokenizer.eos_token_id),
                    )
                    for offset, ids in enumerate(generated):
                        row = {
                            "prompt_id": prompt.prompt_id,
                            "split": prompt.split,
                            "prompt": prompt.prompt,
                            "target": prompt.target,
                            "candidate_index": first + offset,
                            "completion": tokenizer.decode(ids, skip_special_tokens=True),
                            "token_ids": ids,
                            "model_name": settings.model_name,
                            "generation_seed": seed,
                            "temperature": settings.temperature,
                            "top_k": settings.top_k,
                            "max_new_tokens": settings.max_new_tokens,
                        }
                        target.write(json.dumps(row, ensure_ascii=False) + "\n")
                        written += 1
                print(f"cached {settings.candidates_per_prompt} responses: {prompt.prompt_id}")
        tmp_path.rename(settings.output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    print(f"Cached {written} frozen GPT-2 completions in {settings.output_path}")
    return written


def main() -> None:
    build_cache(tyro.cli(Settings))


if __name__ == "__main__":
    main()
