from typing import Protocol


class Tokenizer(Protocol):
    """Minimal tokenizer interface required by the pipeline."""

    def encode(self, text: str) -> list[int]: ...
