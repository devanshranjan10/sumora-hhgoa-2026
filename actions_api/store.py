"""Pluggable Decision-record store (spec section 8).

The actions API writes one Decision record per action call. The default store
keeps them in memory (tests, single-process runs). ``JsonlDecisionStore``
appends to a JSONL file (benchmark audit trail). A TigerGraph-backed store can
be dropped in later — it only has to satisfy ``DecisionStore``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Protocol

from .models import Decision


class DecisionStore(Protocol):
    def write(self, decision: Decision) -> str:
        """Persist a decision; return its id."""
        ...

    def list_for_case(self, case_id: str) -> List[Decision]:
        ...

    def all(self) -> List[Decision]:
        ...


class InMemoryDecisionStore:
    def __init__(self) -> None:
        self._decisions: List[Decision] = []
        self._lock = threading.Lock()

    def write(self, decision: Decision) -> str:
        with self._lock:
            self._decisions.append(decision)
        return decision.decision_id

    def list_for_case(self, case_id: str) -> List[Decision]:
        return [d for d in self._decisions if d.case_id == case_id]

    def all(self) -> List[Decision]:
        return list(self._decisions)

    def clear(self) -> None:
        with self._lock:
            self._decisions.clear()


class JsonlDecisionStore:
    """Append-only JSONL store; doubles as the audit trail."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, decision: Decision) -> str:
        line = json.dumps(decision.model_dump(mode="json"), sort_keys=True)
        with self._lock:
            with self._path.open("a") as fh:
                fh.write(line + "\n")
        return decision.decision_id

    def _read_all(self) -> List[Decision]:
        if not self._path.exists():
            return []
        out: List[Decision] = []
        for line in self._path.read_text().splitlines():
            if line.strip():
                out.append(Decision.model_validate(json.loads(line)))
        return out

    def list_for_case(self, case_id: str) -> List[Decision]:
        return [d for d in self._read_all() if d.case_id == case_id]

    def all(self) -> List[Decision]:
        return self._read_all()
