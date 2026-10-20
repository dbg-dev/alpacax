from __future__ import annotations

from collections.abc import Mapping

from jax import Array, device_get, process_index  # pyright: ignore[reportUnknownVariableType]


def is_lead_process() -> bool:
    return process_index() == 0


def to_host_metrics(metrics: Mapping[str, Array | float]) -> dict[str, float]:
    return {name: float(device_get(value)) for name, value in metrics.items()}
