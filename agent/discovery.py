"""Shipped undocumented typology — loaded from the discovery eval artifact.

The agent never invents a sixth pattern. It only attaches `undocumented` when:
  1. eval/pattern_discovery.json says significant_at_0.05 == True, AND
  2. the case's device fingerprint matches a discovered ring (or the demo
     exemplar), OR the case shares a device with a bank-seeded undocumented
     closed case.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Set

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "eval" / "pattern_discovery.json"


@lru_cache(maxsize=1)
def load_discovery() -> Dict[str, Any]:
    if not ARTIFACT.exists():
        return {"significant_at_0.05": False, "typology": {"shipped": False}}
    return json.loads(ARTIFACT.read_text())


def shipped_typology() -> Optional[Dict[str, str]]:
    d = load_discovery()
    typ = d.get("typology") or {}
    if not (d.get("significant_at_0.05") and typ.get("shipped")):
        return None
    return {
        "pattern": "undocumented",
        "pattern_id": typ.get("pattern_id", "cross_account_device_ring"),
        "name": typ.get("name", ""),
        "pattern_description": (
            "A Build-qualified device profile links confirmed-fraud cases across at least "
            "three distinct cardholders. This cross-account structure differs from a "
            "single-card behavioral burst. A shared profile is an investigative lead, "
            "not proof that one person controlled the accounts."
        ),
    }


def ring_devices() -> Set[str]:
    d = load_discovery()
    if not d.get("significant_at_0.05"):
        return set()
    out: Set[str] = set()
    demo = d.get("demo_exemplar") or {}
    if demo.get("device"):
        out.add(demo["device"])
    for e in d.get("top_exemplars") or []:
        if e.get("device"):
            out.add(e["device"])
    return out


def match_case_device(device: str) -> Optional[Dict[str, Any]]:
    """If this device is in a shipped ring, return the exemplar payload."""
    if not device:
        return None
    typ = shipped_typology()
    if typ is None:
        return None
    d = load_discovery()
    demo = d.get("demo_exemplar") or {}
    if device == demo.get("device"):
        return {**typ, "exemplar": demo}
    for e in d.get("top_exemplars") or []:
        if e.get("device") == device:
            return {**typ, "exemplar": e}
    return None
