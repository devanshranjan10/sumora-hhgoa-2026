from bench.dataset import DatasetIndex
from service.watchlist import build_watchlist


def test_watchlist_uses_only_prior_fraud_and_specific_cross_customer_devices():
    idx = DatasetIndex()
    specific = "Phone Build/ABC | Android 7"
    generic = "Windows | chrome 63.0"
    for number in range(3):
        case_id = f"CC-{number:04d}"
        txn_id = f"T{number:07d}"
        idx.closed_case_txns[case_id] = {txn_id}
        idx.closed_case_outcome_fraud.add(case_id)
        idx.closed_case_closed_at[case_id] = "2016-10-01 00:00:00"
        idx.txn_device[txn_id] = specific
        idx.txn_customer[txn_id] = f"C{number:05d}"
    for txn_id, device, timestamp in [
        ("T1000000", specific, "2016-11-01 00:00:00"),
        ("T1000001", specific, "2016-09-01 00:00:00"),
        ("T1000002", generic, "2016-11-01 00:00:00"),
    ]:
        idx.txn_device[txn_id] = device
        idx.txn_customer[txn_id] = "C90000"
        idx.txn_ts[txn_id] = timestamp
        idx.txn_risk[txn_id] = 0.1
        idx.txn_amount[txn_id] = 100.0

    result = build_watchlist(idx, set())

    assert result["matching_transactions"] == 1
    assert result["items"][0]["transaction_id"] == "T1000000"
    assert result["items"][0]["prior_customer_count"] == 3

    idx.closed_case_closed_at["CC-0002"] = "2016-12-01 00:00:00"
    assert build_watchlist(idx, set())["matching_transactions"] == 0
