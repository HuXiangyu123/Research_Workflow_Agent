from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    label: str
    url: str
    reason: str

