from returns.result import Failure, Success

from alpacax.cli import load
from alpacax.cli.load import CliArgs
from alpacax.types.models import DatasetSplit


def test_load_cli_prints_dataset_summary(monkeypatch, capsys):
    def fake_build_dataset(url, tokenizer, config):
        return Success(
            DatasetSplit(
                train=[object(), object()],
                validation=[object()],
                test=[object()],
                skipped=[Exception("bad row")],
            )
        )

    monkeypatch.setattr(load, "build_dataset", fake_build_dataset)

    exit_code = load.run(CliArgs(url="https://example.com/data.json"))

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "train=2" in captured.out
    assert "validation=1" in captured.out
    assert "test=1" in captured.out
    assert "skipped=1" in captured.out


def test_load_cli_returns_failure_when_pipeline_fails(monkeypatch, capsys):
    def fake_build_dataset(url, tokenizer, config):
        return Failure(ValueError("boom"))

    monkeypatch.setattr(load, "build_dataset", fake_build_dataset)

    exit_code = load.run(CliArgs(url="https://example.com/data.json"))

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "pipeline failed: boom" in captured.out
