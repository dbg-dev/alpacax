from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from alpacax.trainer.loggers import WandbLogger


def test_wandb_logger_reports_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fail_wandb(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        if name == "wandb":
            raise ModuleNotFoundError("No module named 'wandb'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_wandb)

    with pytest.raises(ImportError, match=r"alpacax\[wandb\]"):
        WandbLogger("test")


def test_wandb_logger_imports_on_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []
    def fake_init(**kw: object) -> None:
        calls.append(("init", kw))

    def fake_log(metrics: dict[str, float], step: int) -> None:
        calls.append(("log", (metrics, step)))

    def fake_finish() -> None:
        calls.append(("finish", None))

    fake_wandb = SimpleNamespace(init=fake_init, log=fake_log, finish=fake_finish)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = WandbLogger("test", {"seed": 42})
    logger.log({"loss": 0.2}, step=3)
    logger.close()

    assert calls == [
        ("init", {"project": "test", "config": {"seed": 42}}),
        ("log", ({"loss": 0.2}, 3)),
        ("finish", None),
    ]
