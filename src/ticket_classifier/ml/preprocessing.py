from __future__ import annotations

import re


def normalize_text(value: str) -> str:
    """Normalize support ticket text before vectorization."""
    cleaned = value.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned
