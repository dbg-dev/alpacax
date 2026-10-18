from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Instruction(BaseModel):
    """One Alpaca-format row."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")
    instruction: str = Field(min_length=1)
    input: str = ""
    output: str = Field(min_length=1)


class AlpacaSample(BaseModel):
    prompt: str
    response: str


class TokenizedSample(BaseModel):
    prompt_tokens: list[int]
    response_tokens: list[int]

    @property
    def token_list(self) -> list[int]:
        return self.prompt_tokens + self.response_tokens


class PaddedSample(BaseModel):
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


class DatasetSplit[T, U](BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)
    train: list[T]
    validation: list[T]
    test: list[T]
    skipped: list[U] | None


class SplitConfig(BaseModel):
    train: float = Field(ge=0, le=1)
    validation: float = Field(ge=0, le=1)
    test: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def must_sum_to_one(self):
        if not abs(self.train + self.validation + self.test - 1.0) < 1e-6:
            raise ValueError("splits must sum to 1.0")
        return self
