"""Features available at the flagged transaction's timestamp.

Behavior is measured over the customer's account. The source does not provide
a complete transaction-to-card mapping, so issuer codes cannot stand in for cards.
"""
from math import sqrt

from agent.calibrator import case_features, memory_signal
from agent.memory import device_memory_maps, memory_features, shuffled_maps
from bench.dataset import DatasetIndex, norm_txn_id

FEATURE_CONTRACT = "account-history-asof-v2"


def account_stats(idx: DatasetIndex, t: str) -> dict:
    cutoff = idx.txn_ts[t]
    customer = idx.txn_customer[t]
    history = [x for x in idx.customer_txns.get(customer, []) if idx.txn_ts[x] < cutoff]
    amounts = [abs(idx.amount(x)) for x in history]
    mean = sum(amounts) / len(amounts) if amounts else 0.0
    std = sqrt(sum((a - mean) ** 2 for a in amounts) / len(amounts)) if amounts else 1.0
    device = idx.txn_device.get(t, "")
    return {
        "small_auth_rate": sum(0 < a <= 5 for a in amounts) / len(amounts) if amounts else 0.0,
        "amount_z": (abs(idx.amount(t)) - mean) / (std or 1.0) if amounts else 0.0,
        "device_novelty": int(bool(device) and all(idx.txn_device.get(x) != device for x in history)),
        "night": int(0 <= int(cutoff[11:13]) <= 5),
    }


def feature_vector(idx: DatasetIndex, flagged_txn_id: str, prior_score: float | None = None,
                   memory_mode: str = "full", neutralize_graph: bool = False) -> dict:
    t = norm_txn_id(flagged_txn_id)
    stats = account_stats(idx, t)
    n_fraud = n_cleared = 0
    if not neutralize_graph and memory_mode != "withheld":
        fraud, cleared = device_memory_maps(idx, before=idx.txn_ts[t])
        if memory_mode.startswith("shuffled"):
            seed = int(memory_mode.split(":", 1)[1]) if ":" in memory_mode else 42
            fraud, cleared = shuffled_maps(fraud, cleared, seed=seed)
        devices = {idx.txn_device[t]} if idx.txn_device.get(t) else set()
        n_fraud, n_cleared = memory_features(fraud, cleared, devices)
    if neutralize_graph:
        stats = dict.fromkeys(stats, 0)
    features = case_features(
        small_auth_rate=stats["small_auth_rate"], amount_z=stats["amount_z"],
        online=idx.txn_online.get(t, True), device_novelty=stats["device_novelty"],
        night_txn=stats["night"], prior_score=idx.txn_risk[t] if prior_score is None else prior_score,
        mem_signal_val=memory_signal(n_fraud, n_cleared),
    )
    return {"features": features, "amount_usd": idx.amount(t),
            "mem_fraud": n_fraud, "mem_cleared": n_cleared}
