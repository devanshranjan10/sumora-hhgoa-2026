"""Find exam-period transactions linked to earlier confirmed fraud by device."""

from __future__ import annotations

from collections import defaultdict

from bench.dataset import DatasetIndex


def build_watchlist(idx: DatasetIndex, submission_txns: set[str], limit: int = 20) -> dict:
    prior_by_device: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    customers_by_device: dict[str, set[str]] = defaultdict(set)
    for transaction_id, device in idx.txn_device.items():
        customers_by_device[device].add(idx.txn_customer.get(transaction_id, ""))
    closed_txns: set[str] = set()
    for case_id, transactions in idx.closed_case_txns.items():
        closed_txns.update(transactions)
        if case_id not in idx.closed_case_outcome_fraud:
            continue
        closed_at = idx.closed_case_closed_at[case_id]
        for transaction_id in transactions:
            device = idx.txn_device.get(transaction_id)
            if device:
                prior_by_device[device].append((case_id, closed_at, idx.txn_customer.get(transaction_id, "")))

    candidates = []
    for transaction_id, device in idx.txn_device.items():
        if "Build/" not in device or len(customers_by_device[device]) > 25:
            continue
        if transaction_id in closed_txns or transaction_id in submission_txns:
            continue
        timestamp = idx.txn_ts.get(transaction_id, "")
        if not timestamp.startswith(("2016-11", "2016-12")):
            continue
        customer_id = idx.txn_customer.get(transaction_id, "")
        prior = {(case_id, prior_customer) for case_id, closed_at, prior_customer in prior_by_device.get(device, ())
                 if closed_at < timestamp and prior_customer != customer_id}
        if not prior:
            continue
        if len({prior_customer for _, prior_customer in prior}) < 3:
            continue
        bank_risk = idx.txn_risk.get(transaction_id, 0.0)
        if bank_risk >= 0.5:
            continue
        prior_ids = sorted(case_id for case_id, _ in prior)
        amount = idx.txn_amount.get(transaction_id, 0.0)
        candidates.append({
            "transaction_id": transaction_id,
            "customer_id": customer_id,
            "timestamp": timestamp,
            "amount_usd": round(amount, 2),
            "bank_risk_score": round(bank_risk, 3),
            "prior_confirmed_cases": prior_ids[:5],
            "prior_case_count": len(prior_ids),
            "prior_customer_count": len({prior_customer for _, prior_customer in prior}),
            "device_customer_count": len(customers_by_device[device]),
            "device_profile": device,
        })
    candidates.sort(key=lambda item: (-item["prior_customer_count"], -item["amount_usd"], item["bank_risk_score"]))
    return {"method": "Exam-period online transactions with bank risk below 0.50 sharing a Build/-qualified device profile with confirmed fraud on at least three other customers. Prior cases must have closed before the transaction; profiles seen on more than 25 customers are excluded. This is an alert, not a fraud verdict.",
            "matching_transactions": len(candidates), "items": candidates[:limit]}
