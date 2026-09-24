"""The official MCP result must become the same row envelope as RESTPP."""

import json

import pytest

from agent.mcp_graph import _decode_result


def test_installed_query_response_decodes_rows() -> None:
    payload = {"success": True, "data": {"result": [{"probes": 3, "card_testing_score": 1.0}]}}
    content = "```json\n" + json.dumps(payload) + "\n```\nQuery executed"
    assert _decode_result(content) == payload["data"]["result"]


def test_mcp_error_cannot_be_treated_as_empty_evidence() -> None:
    payload = {"success": False, "error": "query unavailable"}
    with pytest.raises(RuntimeError, match="query unavailable"):
        _decode_result("```json\n" + json.dumps(payload) + "\n```")
