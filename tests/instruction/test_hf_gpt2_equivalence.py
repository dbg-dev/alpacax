"""
Optional HF/local GPT-2 logits equivalence test.

This is intentionally opt-in because the local GPT-2 constructor is project
specific. To run it, set:

    LOCAL_GPT2_FACTORY="some.module:build_local_gpt2"

The factory must accept a Hugging Face GPT2Config and return an uninitialised
local Flax NNX GPT-2 model with matching shapes.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any

import pytest

pytestmark = pytest.mark.integration


def _load_factory() -> Callable[[Any], Any]:
    spec = os.environ.get("LOCAL_GPT2_FACTORY")
    if not spec:
        from flax import nnx

        from alpacax.models.gpt2.config import (
            AttentionConfig,
            DenseMLPConfig,
            EmbeddingConfig,
            GPTConfig,
            TransformerBlockConfig,
        )
        from alpacax.models.gpt2.gpt2 import GPT2

        def build_local_gpt2(config: Any) -> Any:
            d = config.n_embd
            return GPT2(
                GPTConfig(
                    embedding=EmbeddingConfig(
                        vocab_size=config.vocab_size,
                        d_model=d,
                        dropout_rate=0.0,
                    ),
                    block=TransformerBlockConfig(
                        attention=AttentionConfig(
                            d_model=d,
                            n_heads=config.n_head,
                            dropout_rate=0.0,
                        ),
                        mlp=DenseMLPConfig(d_model=d, expansion_factor=4.0),
                        dropout_rate=0.0,
                    ),
                    n_layers=config.n_layer,
                    max_seq_len=config.n_positions,
                ),
                nnx.Rngs(0),
            )

        return build_local_gpt2

    module_name, sep, func_name = spec.partition(":")
    if not sep:
        pytest.fail("LOCAL_GPT2_FACTORY must have form 'module:function'")

    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def test_hf_gpt2_logits_equivalence_after_mapping():
    required = os.environ.get("HF_EQUIV_REQUIRED") == "1"
    if not required and "LOCAL_GPT2_FACTORY" not in os.environ:
        pytest.skip("HF equivalence is opt-in locally; CI sets HF_EQUIV_REQUIRED=1")

    if required:
        import jax.numpy as jnp
        import numpy as np
        import torch

        # CI installs this integration-only dependency for this required test.
        # Importing it dynamically keeps the static-check environment minimal.
        transformers = importlib.import_module("transformers")
    else:
        jnp = pytest.importorskip("jax.numpy")
        np = pytest.importorskip("numpy")
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")

    from alpacax.integrations.huggingface import model_mapper_or_raise
    from alpacax.models.gpt2.hf_mappings import hfgpt2_to_local_gpt2_rules

    build_local_gpt2 = _load_factory()

    # Tiny random model keeps the test fast and avoids network/downloads.
    hf_config = transformers.GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_ctx=16,
        n_embd=16,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    hf_model = transformers.GPT2LMHeadModel(hf_config).eval()
    local_model = build_local_gpt2(hf_config)

    mapped_model = model_mapper_or_raise(
        src_model=hf_model,
        flax_model=local_model,
        rules=hfgpt2_to_local_gpt2_rules(hf_config.n_layer),
        strict=True,
        allow_unused_src=("lm_head.weight",),  # tied to transformer.wte.weight
    )

    token_ids = [[1, 2, 3, 4]]

    with torch.no_grad():
        hf_logits = hf_model(torch.tensor(token_ids)).logits.detach().cpu().numpy()

    local_out = mapped_model(jnp.asarray(token_ids))
    local_logits = getattr(local_out, "logits", local_out)

    np.testing.assert_allclose(np.asarray(local_logits), hf_logits, rtol=1e-4, atol=1e-4)
