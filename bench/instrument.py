"""tool_calls / tokens / latency instrumentation for the benchmark runner."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, List, Tuple


@dataclass
class CaseMetrics:
    """Accumulates per-case instrumentation. Entries in `tool_log` are
    (tool_name, tokens) pairs so the audit trail can show the call sequence."""

    tool_log: List[Tuple[str, int]] = field(default_factory=list)
    tokens: int = 0
    _t0: float = field(default=0.0)

    def tool_call(self, name: str, tokens: int = 0) -> None:
        self.tool_log.append((name, tokens))
        self.tokens += tokens

    def start(self) -> None:
        self._t0 = time.perf_counter()

    @property
    def tool_calls(self) -> int:
        return len(self.tool_log)

    @property
    def latency_s(self) -> float:
        return round(time.perf_counter() - self._t0, 3) if self._t0 else 0.0


@contextmanager
def timed(metrics: CaseMetrics) -> Iterator[None]:
    metrics.start()
    yield
