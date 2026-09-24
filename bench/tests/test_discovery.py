"""Discovery gate: typology ships only when permutation p < 0.05."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "eval" / "pattern_discovery.json"


def test_discovery_artifact_significant():
    assert ARTIFACT.exists(), "run scripts/pattern_discovery.py first"
    data = json.loads(ARTIFACT.read_text())
    assert data["significant_at_0.05"] is True
    assert data["perm_p_value"] < 0.05
    assert data["typology"]["shipped"] is True
    assert data["typology"]["pattern"] == "undocumented"
    assert data["demo_exemplar"]["n_customers"] >= 3
    assert "SM-G935F" in data["demo_exemplar"]["device"]


def test_agent_discovery_loader():
    from agent.discovery import load_discovery, match_case_device, shipped_typology
    load_discovery.cache_clear()
    typ = shipped_typology()
    assert typ is not None
    assert typ["pattern"] == "undocumented"
    data = load_discovery()
    device = data["demo_exemplar"]["device"]
    hit = match_case_device(device)
    assert hit is not None
    assert hit["pattern"] == "undocumented"
    assert match_case_device("not-a-real-device") is None
