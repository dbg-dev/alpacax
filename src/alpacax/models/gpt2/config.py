from collections.abc import Callable
from typing import ClassVar

from jax import Array
from jax.nn import gelu
from pydantic import BaseModel, ConfigDict, Field, model_validator

type ActivationFn = Callable[[Array], Array]


class FrozenModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )


class EmbeddingConfig(FrozenModel):
    vocab_size: int = Field(gt=0)
    d_model: int = Field(gt=0)
    dropout_rate: float = Field(ge=0, lt=1)


class DenseMLPConfig(FrozenModel):
    d_model: int = Field(gt=0)
    expansion_factor: float = Field(default=4.0, gt=0)
    d_hidden: int | None = Field(default=None, gt=0)
    activation_fn: ActivationFn = gelu
    use_bias: bool = True

    @model_validator(mode="after")
    def validate_config(self):
        if self.d_hidden is None:
            hidden_dim = self.expansion_factor * self.d_model
            if not hidden_dim.is_integer():
                raise ValueError(
                    "expansion_factor * d_model must produce an integer hidden dimension"
                )

        return self

    @property
    def hidden_dim(self) -> int:
        """Effective hidden dimension. `d_hidden` overrides `expansion_factor`."""
        if self.d_hidden is not None:
            return self.d_hidden

        return int(self.expansion_factor * self.d_model)


class AttentionConfig(FrozenModel):
    d_model: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    causal: bool = True
    use_bias: bool = True
    dropout_rate: float = Field(ge=0, lt=1)

    @model_validator(mode="after")
    def validate_config(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        return self

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def qkv_dim(self) -> int:
        return 3 * self.d_model


class TransformerBlockConfig(FrozenModel):
    attention: AttentionConfig
    mlp: DenseMLPConfig
    dropout_rate: float = Field(ge=0, lt=1)

    @model_validator(mode="after")
    def validate_config(self):
        if self.attention.d_model != self.mlp.d_model:
            raise ValueError("attention and mlp d_model must match")

        return self

    @property
    def d_model(self) -> int:
        return self.attention.d_model


class GPTConfig(FrozenModel):
    embedding: EmbeddingConfig
    block: TransformerBlockConfig
    n_layers: int = Field(gt=0)
    max_seq_len: int = Field(gt=0)
    tie_embeddings: bool = True

    @model_validator(mode="after")
    def validate_config(self):
        if self.block.d_model != self.embedding.d_model:
            raise ValueError("block and embedding d_model must match")

        return self

    @property
    def d_model(self) -> int:
        return self.embedding.d_model

    @property
    def vocab_size(self) -> int:
        return self.embedding.vocab_size

    @property
    def mlp_hidden_dim(self) -> int:
        return self.block.mlp.hidden_dim

    @property
    def attention_head_dim(self) -> int:
        return self.block.attention.head_dim
