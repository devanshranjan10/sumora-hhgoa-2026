"""Card-level behavior known when a transaction arrives.

The card mapping comes from authoritative supplied cases and card fingerprints.
Every feature uses strictly earlier transactions; missing device history is
represented separately from a novel observed device.
"""
from __future__ import annotations

from bisect import bisect_left
from math import log1p, sqrt

from bench.dataset import DatasetIndex, norm_txn_id

CARD_FEATURE_NAMES = (
    "log_card_history", "card_amount_z", "nearest_amount_rel_90d",
    "similar_amount_count_30d", "similar_amount_count_90d",
    "recent_count_1h", "recent_count_24h", "recent_count_7d",
    "hours_since_previous", "device_observed", "device_seen_before",
    "same_device_count_90d", "same_hour_fraction",
)


class CardHistory:
    def __init__(self, idx: DatasetIndex):
        self.idx = idx
        self.by_card = {
            card: sorted(txns, key=lambda txn: (idx.txn_dt[txn], txn))
            for card, txns in idx.card_txns.items()
        }
        self.times = {
            card: [idx.txn_dt[txn] for txn in txns]
            for card, txns in self.by_card.items()
        }

    def features(self, txn_id: str) -> list[float]:
        txn_id = norm_txn_id(txn_id)
        idx = self.idx
        card = idx.txn_card[txn_id]
        cutoff = idx.txn_dt[txn_id]
        txns = self.by_card.get(card, [])
        times = self.times.get(card, [])
        end = bisect_left(times, cutoff)
        prior = txns[:end]
        recent_90 = [t for t in prior if idx.txn_dt[t] >= cutoff - 90 * 86400]
        amount = abs(idx.amount(txn_id))
        amounts = [abs(idx.amount(t)) for t in prior]
        mean = sum(amounts) / len(amounts) if amounts else amount
        std = sqrt(sum((x - mean) ** 2 for x in amounts) / len(amounts)) if amounts else 1.0
        similar_90 = [t for t in recent_90
                      if abs(abs(idx.amount(t)) - amount) <= max(1.0, .02 * amount)]
        similar_30 = [t for t in similar_90 if idx.txn_dt[t] >= cutoff - 30 * 86400]
        nearest = min((abs(abs(idx.amount(t)) - amount) / max(amount, 1.0)
                       for t in recent_90), default=1.0)
        device = idx.txn_device.get(txn_id, "")
        same_device = [t for t in recent_90 if device and idx.txn_device.get(t) == device]
        hour = int(idx.txn_ts[txn_id][11:13])
        same_hour = sum(min((hour - int(idx.txn_ts[t][11:13])) % 24,
                            (int(idx.txn_ts[t][11:13]) - hour) % 24) <= 2
                        for t in recent_90)
        return [
            log1p(len(prior)),
            max(-5.0, min(15.0, (amount - mean) / (std or 1.0))),
            min(nearest, 5.0),
            log1p(len(similar_30)), log1p(len(similar_90)),
            log1p(sum(idx.txn_dt[t] >= cutoff - 3600 for t in recent_90)),
            log1p(sum(idx.txn_dt[t] >= cutoff - 86400 for t in recent_90)),
            log1p(sum(idx.txn_dt[t] >= cutoff - 7 * 86400 for t in recent_90)),
            min((cutoff - times[end - 1]) / 3600, 2160.0) if end else 2160.0,
            float(bool(device)),
            float(any(device and idx.txn_device.get(t) == device for t in prior)),
            log1p(len(same_device)),
            same_hour / len(recent_90) if recent_90 else 0.0,
        ]
