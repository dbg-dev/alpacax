from typing import Any

import requests
from returns.result import safe

# ── Safe I/O boundaries ───────────────────────────────────────────────────────


@safe
def fetch_raw(url: str, timeout_seconds: float) -> Any:
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()
