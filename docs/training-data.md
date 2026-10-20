# Instruction pipeline to trainer bridge

The core pipeline returns a `DatasetSplit[PaddedSample, DatasetError]`,
while the generic trainer accepts any iterable of `(inputs, labels)` JAX
arrays. `PaddedBatchLoader` supplies a small **in-memory, re-iterable** bridge
without imposing Grain on the trainer or importing JAX from the core pipeline.

Install the training dependencies with `pip install 'alpacax[train]'`.

```python
from functools import partial

from returns.result import Failure

from alpacax.pipelines.instruction.config import PipelineConfig
from alpacax.pipelines.instruction.runner import build_dataset
from alpacax.trainer.core import run
from alpacax.trainer.data import PaddedBatchLoader
from alpacax.trainer.factories import make_runner, make_train_phase
from alpacax.trainer.losses import masked_cross_entropy_loss

config = PipelineConfig(eos_token_id=your_eos_id)
result = build_dataset(dataset_url, tokenizer, config)
if isinstance(result, Failure):
    raise result.failure()

split = result.unwrap()
loader = PaddedBatchLoader(split.train, batch_size=8, shuffle=True, seed=config.seed)
train = make_train_phase(
    model=model,
    optimizer=optimizer,
    loader=loader,
    metrics=metrics,
    loss_fn=partial(masked_cross_entropy_loss, ignore_index=config.ignore_index),
)
run(make_runner(train=train, max_steps=100))
```

In this example, `dataset_url`, `tokenizer`, `your_eos_id`, `model`,
`optimizer` and `metrics` must be supplied by the caller. The model's
vocabulary must contain all token IDs produced by the tokenizer.

## Unpadded records with bucketed batch-time padding

Use `build_unpadded_dataset` when you want to avoid eagerly padding all
records to `max_seq_len - 1`. It produces `PreparedSample` records containing
only shifted `input_ids` and prompt-masked `labels`. The trainer's generic
`(x, y)` contract remains unchanged.

```python
from alpacax.pipelines.instruction.runner import build_unpadded_dataset
from alpacax.trainer.data import BucketedBatchLoader

compact_result = build_unpadded_dataset(dataset_url, tokenizer, config)
if isinstance(compact_result, Failure):
    raise compact_result.failure()

loader = BucketedBatchLoader(
    compact_result.unwrap().train,
    batch_size=8,
    bucket_lengths=(128, 256, 512, 1024, 2048),
    pad_token=config.pad_token,
    ignore_index=config.ignore_index,
    shuffle=True,
    seed=config.seed,
)
# Pass loader directly to make_train_phase(loader=loader, ...)
```

The `bucket_lengths` refer to **shifted input sequence lengths**, not original
input+label token counts. For each record the loader selects the smallest
bucket it fits, groups records into batches within that bucket and right-pads
only that batch to the bucket's length. `max_seq_len` remains an upper bound on
the original token sequence, not an instruction to pad every record eagerly.
Any record longer than the largest configured bucket is rejected explicitly.

Shuffling is repeatable for a fixed seed on each iteration. Construct a new
loader with a different seed to change ordering between epochs.
`drop_last=True` removes partial batches **per bucket**, not just one final
batch overall, and provides a fixed batch dimension for JAX compilation.
With `drop_last=False`, the smaller batches can introduce extra compilation
shapes. Bucket count determines the number of distinct sequence dimensions.

This is still an **in-memory** pipeline of Python integer lists; it reduces
eager padding and unnecessary training compute but does not yet stream rows
from the source or pack separate training sequences.

## Semantics and limitations

- Inputs and labels have **already been shifted** by `prepare_sample`.
  Do not shift them again.
- Prompt and padding labels are set to `config.ignore_index`; use the same
  `ignore_index` in the loss. Use `masked_cross_entropy_loss` rather than
  unmasked `cross_entropy_loss`.
- The pipeline right-pads inputs and masks padding labels. Causal attention
  prevents real positions from attending to future padding, so the separate
  `attention_mask` need not be passed for this **right-padded, unpacked**
  causal-LM setup. Left padding and packed sequences require additional
  attention masking, not supported by this adapter.
- The loader retains the supplied sample sequence in memory. Batches are
  converted to int32 JAX arrays when iterated; the last batch may be smaller
  unless `drop_last=True`. It does not prefetch or place batches on devices
  explicitly.
- This loader is **re-iterable**, which is important for repeated evaluation;
  it performs the same deterministic shuffle on each pass for a given seed.
- `build_dataset` retains the original eager padding for compatibility;
  `build_unpadded_dataset` and `BucketedBatchLoader` provide the compact
  alternative. Streaming and segment-aware packing are future features.
- Metrics consuming raw labels must also handle ignored targets correctly.
