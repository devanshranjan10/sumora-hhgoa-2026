"""Test 1: the README's own example answer must validate against our schema."""

import json
import re
from pathlib import Path

import pytest

from bench.validate import validate_schema, validate_semantics

README = Path(__file__).resolve().parents[2] / "data" / "HHGOA_IEEE" / "README.md"


@pytest.fixture(scope="module")
def readme_example():
    """Extracted verbatim from the README 'Example' section at collection time,
    so the test always tracks the authoritative document."""
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "readme_example_answer.json"
    if not README.exists():
        return json.loads(fixture.read_text())
    text = README.read_text()
    m = re.search(r"### Example\s+```json\s+(\{.*?\n\})\s*```", text, re.S)
    assert m, "could not extract example JSON from README"
    return json.loads(m.group(1))


def test_example_matches_extracted_fixture(readme_example):
    if not README.exists():
        pytest.skip("the challenge README is not included in Git")
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "readme_example_answer.json"
    assert json.loads(fixture.read_text()) == readme_example


def test_example_passes_schema(readme_example):
    errors = validate_schema(readme_example)
    assert errors == [], f"README example failed schema: {errors}"


def test_example_passes_semantics(readme_example):
    errors = validate_semantics(readme_example)
    assert errors == [], f"README example failed semantic checks: {errors}"


def test_schema_rejects_missing_field(readme_example):
    broken = dict(readme_example)
    del broken["stop_reason"]
    assert validate_schema(broken) != []


def test_schema_rejects_sar_without_file_report(readme_example):
    broken = json.loads(json.dumps(readme_example))
    broken["next_best_actions"]["final"] = [
        a for a in broken["next_best_actions"]["final"] if a["action"] != "FILE_REPORT"
    ]
    # schema alone passes; the semantic cross-field rule must catch it
    assert validate_schema(broken) == []
    assert any("sar.file" in e for e in validate_semantics(broken))


def test_schema_rejects_unknown_action(readme_example):
    broken = json.loads(json.dumps(readme_example))
    broken["next_best_actions"]["final"][0]["action"] = "LAUNCH_ROCKET"
    assert validate_schema(broken) != []
