import csv

from bench.dataset import load_dataset


def test_dataset_uses_supplied_identity_calendar_channel_and_risk(tmp_path):
    tables = {
        "transactions.csv": [
            {"TransactionID": "3000003", "TransactionDT": "187", "TransactionAmt": "12.5",
             "card1": "22374", "customer_id": "C11919", "ts": "2016-07-02 00:03:07",
             "channel": "in_person", "ProductCD": "W", "risk_score": "0.52"},
        ],
        "identity.csv": [{"TransactionID": "3000003", "DeviceInfo": "device"}],
        "closed_cases_history.csv": [
            {"case_id": "CC-0001", "card_id": "C11919-K2", "customer_id": "C11919",
             "txn_ids": "3000003", "outcome": "cleared",
             "opened_at": "2016-07-02 01:00:00", "closed_at": "2016-07-04 01:00:00"},
        ],
    }
    for name, rows in tables.items():
        with (tmp_path / name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    idx = load_dataset(tmp_path)
    assert idx.txn_card["T3000003"] == "C11919-K2"
    assert idx.date_of("T3000003") == "2016-07-02"
    assert idx.txn_online["T3000003"] is False
    assert idx.txn_risk["T3000003"] == 0.52
    assert idx.closed_case_cards["CC-0001"] == {"C11919-K2"}
