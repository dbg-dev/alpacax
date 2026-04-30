import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from returns.result import Success

from alpacax.pipelines.instruction.config import PipelineConfig
from alpacax.pipelines.instruction.runner import build_dataset


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


def serve_directory(directory: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, *args, **kwargs):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_build_dataset_integration_success(tmp_path):
    dataset = [
        {
            "instruction": "Say hello",
            "input": "",
            "output": "Hello!",
        },
        {
            "instruction": "Say bye",
            "input": "",
            "output": "Bye!",
        },
    ]

    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    server = serve_directory(tmp_path)

    try:
        url = f"http://127.0.0.1:{server.server_port}/dataset.json"

        config = PipelineConfig(
            seed=123,
            request_timeout_seconds=2,
        )

        result = build_dataset(
            url=url,
            tokenizer=FakeTokenizer(),
            config=config,
        )

    finally:
        server.shutdown()
        server.server_close()

    assert isinstance(result, Success)

    split = result.unwrap()

    assert len(split.train) + len(split.validation) + len(split.test) == 2