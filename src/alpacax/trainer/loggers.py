from collections.abc import Mapping
from typing import Any, Protocol, cast


class ConsoleLogger:
    def log(self, metrics: dict[str, float], step: int) -> None:
        values = " ".join(f"{name}: {value:.4f}" for name, value in metrics.items())
        print(f"step: {step} {values}")

    def close(self) -> None:
        pass


class _WandbAPI(Protocol):
    def init(self, *, project: str, config: dict[str, Any]) -> object: ...
    def log(self, metrics: dict[str, float], *, step: int) -> None: ...
    def finish(self) -> None: ...


class WandbLogger:
    def __init__(self, project: str, config: Mapping[str, Any] | None = None):
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "WandbLogger requires W&B. Install it with "
                "`pip install 'alpacax[wandb]'` or `uv sync --extra wandb`."
            ) from exc

        self._wandb: _WandbAPI = cast(_WandbAPI, cast(object, wandb))
        self._wandb.init(project=project, config=dict(config or {}))

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def close(self) -> None:
        self._wandb.finish()
