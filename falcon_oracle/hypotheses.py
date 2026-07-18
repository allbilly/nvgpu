"""Machine-readable hypothesis board from progress.md (offline)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_BOARD = (
    Path(__file__).resolve().parent / "corpus" / "bringup" / "hypotheses.json"
)


@dataclass
class Hypothesis:
    id: str
    status: str
    oracle: str
    summary: str


def load_hypotheses(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_BOARD
    return json.loads(p.read_text())


def list_hypotheses(path: Path | str | None = None) -> list[Hypothesis]:
    data = load_hypotheses(path)
    return [
        Hypothesis(
            id=h["id"],
            status=h["status"],
            oracle=h["oracle"],
            summary=h["summary"],
        )
        for h in data.get("hypotheses") or []
    ]


def format_hypotheses(path: Path | str | None = None) -> str:
    data = load_hypotheses(path)
    lines = [
        "Kepler hypothesis board (progress.md, offline)",
        f"  leading: {', '.join(data.get('leading') or [])}",
        "  entries:",
    ]
    for h in list_hypotheses(path):
        lines.append(
            f"    [{h.oracle:12}] {h.id:4} ({h.status}): {h.summary}"
        )
    lines.append("  next cheap live discriminator: H79 nested 8fe8 stop 0xa2ba (not this tool)")
    return "\n".join(lines)
