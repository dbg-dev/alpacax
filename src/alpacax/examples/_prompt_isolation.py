"""Prompt-isolation check shared by the Gao pilot's prompt and cache loaders."""

from __future__ import annotations

from collections.abc import Iterable

_MAX_REPORTED = 5


def normalize_prompt(text: str) -> str:
    """Case-fold and collapse whitespace so trivial variants compare equal."""
    return " ".join(text.casefold().split())


def check_prompt_isolation(entries: Iterable[tuple[str, str, str]]) -> None:
    """Raise ValueError if one prompt text appears in more than one split.

    Each entry is ``(prompt_id, split, prompt_text)``. Distinct prompt IDs are
    not enough to keep splits independent: the same text under two IDs would
    let a held-out prompt leak into training. This detects exact duplicates
    after case-folding and whitespace normalization. Near-duplicates and
    paraphrases are not detected. Repeats inside a single split are allowed.
    """
    first_seen: dict[str, tuple[str, str]] = {}
    clashes: list[str] = []
    for prompt_id, split, text in entries:
        first_id, first_split = first_seen.setdefault(normalize_prompt(text), (prompt_id, split))
        if first_split != split:
            clashes.append(f"{first_id!r} ({first_split}) and {prompt_id!r} ({split})")

    if clashes:
        shown = "; ".join(clashes[:_MAX_REPORTED])
        hidden = len(clashes) - _MAX_REPORTED
        suffix = f" (+{hidden} more)" if hidden > 0 else ""
        raise ValueError(f"Prompt text appears in more than one split: {shown}{suffix}")
