from collections.abc import Mapping
from typing import Any


class ConsoleLogger:
    def log(self, metrics: dict[str, float], step: int) -> None:
        values = " ".join(f"{name}: {value:.4f}" for name, value in metrics.items())
        print(f"step: {step} {values}")

    def close(self) -> None:
        pass


class WandbLogger:
    def __init__(self, project: str, config: Mapping[str, Any] | None = None):
        import wandb

        self._wandb = wandb
        self._wandb.init(project=project, config=dict(config or {}))

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def close(self) -> None:
        self._wandb.finish()
