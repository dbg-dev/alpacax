from __future__ import annotations

from alpacax.models.gpt2.hf_mappings import (
    AUTO,
    IDENTITY,
    GPT2LocalKeys,
    hf,
    hfgpt2_to_local_gpt2_rules,
    local,
    rule,
)
from alpacax.params.mapping import MappingRule, TransformKind


def test_gpt2_rules_have_expected_count():
    rules = hfgpt2_to_local_gpt2_rules(num_layers=2)

    assert len(rules) == 2 * 12 + 4


def test_gpt2_rules_use_explicit_identity_transform():
    rules = hfgpt2_to_local_gpt2_rules(num_layers=1)

    assert all(rule.transform is IDENTITY for rule in rules)


def test_gpt2_rules_use_named_local_keys():
    rules = hfgpt2_to_local_gpt2_rules(num_layers=1)

    assert rules[3].dst == local.mha_qkv_kernel(0)
    assert rules[3].src == hf.attn_qkv_weight(0)


def test_rule_dsl_is_just_a_mapping_rule():
    act = rule(
        dst=local.mha_qkv_kernel(0),
        src=hf.attn_qkv_weight(0),
        transform=AUTO,
    )

    assert act == MappingRule(
        dst=GPT2LocalKeys.mha_qkv_kernel(0),
        src="transformer.h.0.attn.c_attn.weight",
        transform=TransformKind.AUTO,
    )
