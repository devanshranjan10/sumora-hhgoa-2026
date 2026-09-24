import csv

from bench.card_identity import learn_card_fingerprints, resolve_card


def _write(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def test_authoritative_card_links_override_customer_and_missing_card2(tmp_path):
    rows = [
        {"TransactionID": "3000001", "customer_id": "C00001", "card1": "42",
         "card2": "500", "card6": "credit"},
        {"TransactionID": "3000002", "customer_id": "C00001", "card1": "42",
         "card2": "500", "card6": "debit"},
        {"TransactionID": "3000003", "customer_id": "C00001", "card1": "42",
         "card2": "500", "card6": "debit"},
        {"TransactionID": "3000004", "customer_id": "C00001", "card1": "42",
         "card2": "", "card6": "debit"},
    ]
    _write(tmp_path / "transactions.csv", rows)
    _write(tmp_path / "closed_cases_history.csv", [
        {"case_id": "CC-0001", "txn_ids": "3000001", "card_id": "C00001-K1"},
        {"case_id": "CC-0002", "txn_ids": "3000002", "card_id": "C00001-K2"},
    ])
    mapping, links = learn_card_fingerprints(tmp_path)
    assert resolve_card(rows[0], mapping, links) == "C00001-K1"
    assert resolve_card(rows[1], mapping, links) == "C00001-K2"
    assert resolve_card(rows[2], mapping, links) == "C00001-K2"
    assert resolve_card(rows[3], mapping, links).startswith("C00001-U")
