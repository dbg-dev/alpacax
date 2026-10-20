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
- Preparing all samples as Python lists and padding them to `max_seq_len - 1`
  is still expensive. A subsequent change should introduce unpadded prepared
  samples with bucketed, batch-time padding. Streaming and packing come later.
- Metrics consuming raw labels must also handle ignored targets correctly.
