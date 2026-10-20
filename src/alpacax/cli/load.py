from dataclasses import dataclass

import tyro
from returns.result import Success

from alpacax.pipelines.instruction.config import PipelineConfig
from alpacax.pipelines.instruction.runner import build_dataset


class DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]


@dataclass(frozen=True)
class CliArgs:
    url: str
    max_seq_len: int = 512
    seed: int = 42


def run(args: CliArgs) -> int:
    result = build_dataset(
        args.url,
        DummyTokenizer(),
        PipelineConfig(
            max_seq_len=args.max_seq_len,
            truncation="truncate",
            eos_token_id=0,
            pad_token=0,
            seed=args.seed,
        ),
    )

    if isinstance(result, Success):
        split = result.unwrap()
        print(f"train={len(split.train)}")
        print(f"validation={len(split.validation)}")
        print(f"test={len(split.test)}")
        print(f"skipped={len(split.skipped or [])}")
        return 0

    print(f"pipeline failed: {result.failure()}")
    return 1


def main() -> int:
    return run(tyro.cli(CliArgs))


if __name__ == "__main__":
    raise SystemExit(main())
