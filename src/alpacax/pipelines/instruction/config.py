import warnings
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alpacax.types.models import SplitConfig


class PipelineConfig(BaseModel):
    """Runtime configuration for dataset construction."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    weights: SplitConfig = SplitConfig(train=0.8, validation=0.1, test=0.1)
    max_seq_len: int = Field(default=2048, ge=2)
    min_seq_len: int = Field(default=2, ge=2)
    truncation: Literal["skip", "truncate"] = "skip"
    pad_token: int = 0
    ignore_index: int = -100
    seed: int = 42
    mask_prompt_labels: bool = True
    add_eos_token: bool = True
    eos_token_id: int | None = None
    request_timeout_seconds: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def validate_config(self):
        if self.min_seq_len > self.max_seq_len:
            raise ValueError("min_seq_len cannot exceed max_seq_len")
        if self.add_eos_token and self.eos_token_id is None:
            # This is allowed because some tokenizers include EOS in encode().
            warnings.warn(
                "No eos token id is provided. The pipeline will not append EOS unless an id is provided",
                stacklevel=2,
            )
            return self
        return self
