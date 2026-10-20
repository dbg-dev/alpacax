# Strict parameter-mapping coverage

The generic mapper (`alpacax.params.mapping.map_param_dict`) and the
Hugging Face/Flax NNX adapters (`model_mapper`, `model_mapper_or_raise`)
default to **permissive** mapping: keys not mentioned in the mapping rules are
allowed, and unmapped destination values retain their initialization.

Use `strict=True` to verify complete key coverage when importing model
weights for inference, fine-tuning, or equivalence testing.

```python
from alpacax.integrations.huggingface import model_mapper_or_raise
from alpacax.models.gpt2.hf_mappings import hfgpt2_to_local_gpt2_rules

mapped = model_mapper_or_raise(
    src_model=hf_model,
    flax_model=local_model,
    rules=hfgpt2_to_local_gpt2_rules(hf_config.n_layer),
    strict=True,
    allow_unused_src=("lm_head.weight",),
)
```

## Coverage semantics

In strict mode, errors are returned for:

- **`unmapped_dst`**: a destination array key not present in any mapping rule.
- **`unused_src`**: a source parameter key not referenced by any mapping rule.

These errors use the existing `MappingError` results and human-readable
formatting helpers. Rule-level errors (missing keys, incompatible shapes,
ambiguous transforms) are reported together with coverage failures.

Coverage checks use **declared rule keys**, not distinct underlying tensor
values. One source may legitimately map to several destinations. A declared
but invalid rule still fails its existing resolution/shape validation.

For the NNX adapter, only **JAX-array-valued `nnx.Param` leaves** count as
required destination targets. The adapter also carries array-valued NNX
`RngState` (random keys/counters) and other variables, but does not require
Hugging Face mapping rules for them. Non-parameter state is preserved.

### Intentional exceptions

`allow_unused_src` lists source keys intentionally not imported, for example
the `lm_head.weight` alias when using HF GPT-2's tied embedding/output weights.
`allow_unmapped_dst` lists destination array keys that should intentionally
retain their current values.

Prefer short, reviewed allowlists over weakening strict mode globally.
A name on an exception list is excluded from the corresponding strict coverage
check, **not** from any explicit rule; mapping rules are still validated and
applied normally.

For the generic mapper:

```python
result = map_param_dict(
    src=source_arrays,
    dst=target_arrays,
    rules=rules,
    convert=convert,
    transpose_last_two=transpose_last_two,
    strict=True,
    allow_unmapped_dst=(optional_target_key,),
    allow_unused_src=(tied_source_alias,),
)
```

## Failure behavior

Mapping validation and strict coverage finish before values are converted or
applied. A failed check returns `Failure(list[MappingError])` from
`map_param_dict`/`model_mapper`. `model_mapper_or_raise` raises a
`ValueError` containing formatted errors. No partially mapped model is
returned after a strict coverage failure.

Strict mode does **not** validate value semantics (e.g. a shape-compatible
but incorrectly named source parameter). It should complement, not replace,
tests comparing model outputs before and after import.
