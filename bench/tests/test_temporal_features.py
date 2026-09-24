from agent.tools import InvestigationTools
from bench.dataset import DatasetIndex


def test_features_ignore_future_transactions_and_unresolved_case_outcomes():
    idx = DatasetIndex(
        txn_ids={"T0000001", "T0000002"},
        txn_amount={"T0000001": 2, "T0000002": 100},
        txn_ts={"T0000001": "2016-08-01 12:00:00", "T0000002": "2016-08-02 12:00:00"},
        txn_dt={"T0000001": 1, "T0000002": 2},
        txn_customer={"T0000001": "C00001", "T0000002": "C00001"},
        customer_txns={"C00001": ["T0000001", "T0000002"]},
        card_txns={"C00001-K1": ["T0000001", "T0000002"]},
        txn_device={"T0000001": "old", "T0000002": "new"},
    )
    tools = InvestigationTools(idx)
    before = tools.q_case_feature_vector("T0000002", "C00001-K1", "C00001", 0.63).data["features"]
    idx.txn_ids.add("T0000003")
    idx.txn_amount["T0000003"] = 10000
    idx.txn_ts["T0000003"] = "2016-09-01 00:00:00"
    idx.txn_dt["T0000003"] = 3
    idx.txn_device["T0000003"] = "new"
    idx.customer_txns["C00001"].append("T0000003")
    idx.card_txns["C00001-K1"].append("T0000003")
    idx.closed_case_ids.add("CC-0001")
    idx.closed_case_outcome_fraud.add("CC-0001")
    idx.closed_case_txns["CC-0001"] = {"T0000003"}
    idx.closed_case_closed_at["CC-0001"] = "2016-09-02 00:00:00"
    after = InvestigationTools(idx).q_case_feature_vector("T0000002", "C00001-K1", "C00001", 0.63).data["features"]
    assert after == before
    assert after[3] == 1
    assert after[5] == 0.63
    assert after[6] == 0
