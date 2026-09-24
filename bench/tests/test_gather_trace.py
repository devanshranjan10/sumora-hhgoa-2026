
from bench.agent import CaseContext
from bench.dataset import load_dataset, DATA_DIR
from bench.instrument import CaseMetrics
from agent.runner import GraphAgentRunner


def test_non_customer_gather_is_recorded_in_answer(tmp_path, monkeypatch):
    # The actual benchmark has a step-up gather on HHG-019 under the selected
    # model. This test checks the public answer contract, not private nodes.
    monkeypatch.setenv("SUMORA_TRACE_DIR", str(tmp_path / "traces"))
    row = next(r for r in __import__("csv").DictReader((DATA_DIR / "case_pack.csv").open())
               if r["case_id"] == "HHG-019")
    ctx = CaseContext(case_id=row["case_id"], opened_at=row["opened_at"],
                      trigger_type=row["trigger_type"], trigger_text=row["trigger_text"],
                      flagged_txn_id=row["flagged_txn_id"], card_id=row["card_id"],
                      customer_id=row["customer_id"], risk_score=float(row["risk_score"]))
    ans = GraphAgentRunner(seed=42).investigate(ctx, load_dataset(), CaseMetrics()).answer
    if any(r["type"] != "customer_validation" for r in ans["evidence_requests"]):
        assert ans["next_best_actions"]["what_changed"] != "nothing"
