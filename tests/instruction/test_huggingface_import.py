"""The adapter's import boundary should not require a live PyTorch import."""

from __future__ import annotations

import subprocess
import sys


def test_huggingface_adapter_does_not_import_torch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import alpacax.integrations.huggingface; "
            "assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_hooks_import_does_not_import_checkpoint_module() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import alpacax.trainer.hooks; "
            "assert 'alpacax.trainer.checkpoints' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
