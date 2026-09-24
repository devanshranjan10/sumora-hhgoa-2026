"""Answer-file writer: validate first, then write canonical bytes (spec section 6a)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench.dataset import DatasetIndex
from bench.validate import validate_answer


def canonical_bytes(answer: Dict[str, Any]) -> bytes:
    """Stable serialization so --deterministic runs are byte-identical."""
    return (json.dumps(answer, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_answer(
    answer: Dict[str, Any],
    out_dir: Path,
    idx: Optional[DatasetIndex] = None,
    case_state: Optional[Dict[str, Any]] = None,
    strict: bool = True,
) -> List[str]:
    """Validate and write <case_id>.json. Returns validation errors."""
    errors = validate_answer(answer, idx=idx, case_state=case_state)
    if errors and strict:
        raise ValueError(
            f"answer for {answer.get('case_id', '?')} failed validation:\n  - "
            + "\n  - ".join(errors)
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{answer['case_id']}.json").write_bytes(canonical_bytes(answer))
    return errors
