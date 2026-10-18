"""Model-specific mapping rules for HF GPT-2 -> local GPT-2."""

from __future__ import annotations

from collections.abc import Hashable

from alpacax.params.mapping import AUTO, MappingRule, rule

type ModelParamKey = tuple[Hashable, ...]
type HFParamKey = str

__all__ = [
    "GPT2LocalKeys",
    "GPT2HFKeys",
    "local",
    "hf",
    "AUTO",
    "rule",
    "hfgpt2_to_local_gpt2_rules",
]


class GPT2LocalKeys:
    """Local GPT-2 key helpers.

    These helpers keep the mapping readable and isolate fragile tuple paths in
    one place. If the local model internals change, update this class first.
    """

    @staticmethod
    def mha_norm_bias(i: int) -> ModelParamKey:
        return ("blocks", i, "mha_norm", "bias")

    @staticmethod
    def mha_norm_scale(i: int) -> ModelParamKey:
        return ("blocks", i, "mha_norm", "scale")

    @staticmethod
    def mha_qkv_bias(i: int) -> ModelParamKey:
        return ("blocks", i, "mha", "qkv", "bias")

    @staticmethod
    def mha_qkv_kernel(i: int) -> ModelParamKey:
        return ("blocks", i, "mha", "qkv", "kernel")

    @staticmethod
    def mha_out_bias(i: int) -> ModelParamKey:
        return ("blocks", i, "mha", "out_proj", "bias")

    @staticmethod
    def mha_out_kernel(i: int) -> ModelParamKey:
        return ("blocks", i, "mha", "out_proj", "kernel")

    @staticmethod
    def mlp_norm_bias(i: int) -> ModelParamKey:
        return ("blocks", i, "mlp_norm", "bias")

    @staticmethod
    def mlp_norm_scale(i: int) -> ModelParamKey:
        return ("blocks", i, "mlp_norm", "scale")

    @staticmethod
    def mlp_fc1_bias(i: int) -> ModelParamKey:
        return ("blocks", i, "mlp", "fc1", "bias")

    @staticmethod
    def mlp_fc1_kernel(i: int) -> ModelParamKey:
        return ("blocks", i, "mlp", "fc1", "kernel")

    @staticmethod
    def mlp_fc2_bias(i: int) -> ModelParamKey:
        return ("blocks", i, "mlp", "fc2", "bias")

    @staticmethod
    def mlp_fc2_kernel(i: int) -> ModelParamKey:
        return ("blocks", i, "mlp", "fc2", "kernel")

    @staticmethod
    def tok_embedding() -> ModelParamKey:
        return ("token_embedding", "embedding")

    @staticmethod
    def pos_embedding() -> ModelParamKey:
        return ("position_embedding", "embedding")

    @staticmethod
    def final_ln_bias() -> ModelParamKey:
        return ("layernorm", "bias")

    @staticmethod
    def final_ln_scale() -> ModelParamKey:
        return ("layernorm", "scale")


class GPT2HFKeys:
    @staticmethod
    def ln1_bias(i: int) -> HFParamKey:
        return f"transformer.h.{i}.ln_1.bias"

    @staticmethod
    def ln1_weight(i: int) -> HFParamKey:
        return f"transformer.h.{i}.ln_1.weight"

    @staticmethod
    def attn_qkv_bias(i: int) -> HFParamKey:
        return f"transformer.h.{i}.attn.c_attn.bias"

    @staticmethod
    def attn_qkv_weight(i: int) -> HFParamKey:
        return f"transformer.h.{i}.attn.c_attn.weight"

    @staticmethod
    def attn_out_bias(i: int) -> HFParamKey:
        return f"transformer.h.{i}.attn.c_proj.bias"

    @staticmethod
    def attn_out_weight(i: int) -> HFParamKey:
        return f"transformer.h.{i}.attn.c_proj.weight"

    @staticmethod
    def ln2_bias(i: int) -> HFParamKey:
        return f"transformer.h.{i}.ln_2.bias"

    @staticmethod
    def ln2_weight(i: int) -> HFParamKey:
        return f"transformer.h.{i}.ln_2.weight"

    @staticmethod
    def mlp_fc_bias(i: int) -> HFParamKey:
        return f"transformer.h.{i}.mlp.c_fc.bias"

    @staticmethod
    def mlp_fc_weight(i: int) -> HFParamKey:
        return f"transformer.h.{i}.mlp.c_fc.weight"

    @staticmethod
    def mlp_proj_bias(i: int) -> HFParamKey:
        return f"transformer.h.{i}.mlp.c_proj.bias"

    @staticmethod
    def mlp_proj_weight(i: int) -> HFParamKey:
        return f"transformer.h.{i}.mlp.c_proj.weight"


# Shorthand aliases, purely to keep the rule definitions compact and domain-like.
local = GPT2LocalKeys
hf = GPT2HFKeys


def hfgpt2_to_local_gpt2_rules(
    num_layers: int,
) -> list[MappingRule[ModelParamKey, HFParamKey]]:
    rules: list[MappingRule[ModelParamKey, HFParamKey]] = []

    for i in range(num_layers):
        rules.extend(
            [
                rule(local.mha_norm_bias(i), hf.ln1_bias(i)),
                rule(local.mha_norm_scale(i), hf.ln1_weight(i)),
                rule(local.mha_qkv_bias(i), hf.attn_qkv_bias(i)),
                rule(local.mha_qkv_kernel(i), hf.attn_qkv_weight(i)),
                rule(local.mha_out_bias(i), hf.attn_out_bias(i)),
                rule(local.mha_out_kernel(i), hf.attn_out_weight(i)),
                rule(local.mlp_norm_bias(i), hf.ln2_bias(i)),
                rule(local.mlp_norm_scale(i), hf.ln2_weight(i)),
                rule(local.mlp_fc1_bias(i), hf.mlp_fc_bias(i)),
                rule(local.mlp_fc1_kernel(i), hf.mlp_fc_weight(i)),
                rule(local.mlp_fc2_bias(i), hf.mlp_proj_bias(i)),
                rule(local.mlp_fc2_kernel(i), hf.mlp_proj_weight(i)),
            ]
        )

    rules.extend(
        [
            rule(local.tok_embedding(), "transformer.wte.weight"),
            rule(local.pos_embedding(), "transformer.wpe.weight"),
            rule(local.final_ln_bias(), "transformer.ln_f.bias"),
            rule(local.final_ln_scale(), "transformer.ln_f.weight"),
        ]
    )

    return rules
