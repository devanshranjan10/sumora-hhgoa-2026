from bench.dataset import DatasetIndex
from agent.tools import InvestigationTools


def test_live_graph_uses_raw_transaction_ids(monkeypatch):
    sent = []
    mcp_sent = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": []}

    def post(url, json, timeout):
        sent.append(json)
        return Response()

    monkeypatch.setattr("agent.tools.httpx.post", post)
    def run_query(name, params):
        mcp_sent.append((name, params))
        return {"results": []}

    monkeypatch.setattr("agent.mcp_graph.run_installed_query", run_query)
    tool = InvestigationTools(DatasetIndex(), mode="live")
    tool._gsql("q_new_device_cnp_score", {"txn_id": "T3450629"})
    tool._gsql("q_record_case", {"affected_txn_ids": ["T3450629", "T0345629"]})
    assert mcp_sent == [("q_new_device_cnp_score", {"txn_id": "3450629"})]
    assert tool.mcp_calls[0]["query_name"] == "q_new_device_cnp_score"
    assert tool.mcp_calls[0]["result_preview"] == "[]"
    assert sent[0]["affected_txn_ids"] == ["3450629", "345629"]


def test_absent_graph_device_does_not_become_new_device(monkeypatch):
    tool = InvestigationTools(DatasetIndex(txn_online={"T3450629": True}), mode="live")
    monkeypatch.setattr(tool, "_rows", lambda name, params: [{
        "new_device_cnp_score": 0, "w_device_novel": 0,
        "prior_txns_same_device": 0, "support_count": 1,
    }])
    result = tool.q_new_device_cnp_score("T3450629")
    assert result.data["new_device"] is False
    assert result.data["score"] == 0


def test_structural_similarity_ranks_graph_overlap(monkeypatch):
    tool = InvestigationTools(DatasetIndex(closed_case_outcome_fraud={"CC-0002"}),
                              mode="live")
    monkeypatch.setattr(tool, "_rows", lambda name, params: [{
        "similar_closed_cases": ["CC-0001|2|0", "CC-0002|7|0", "CC-0003|4|0"],
        "similar_cases": ["HHG-003|1|0", "HHG-004|6|0"],
    }])
    result = tool.q_case_structural_similarity("", limit=2, case_id="HHG-001")
    assert result.data["similar_prior_cases"] == ["CC-0002", "CC-0003"]
    assert result.data["agent_cases"] == ["HHG-004", "HHG-003"]
    assert result.data["n_fraud"] == 1
