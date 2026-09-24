"""Case memory (spec §6): signed similarity features from bank-confirmed cases.

mem_fraud / mem_cleared = how many prior bank-confirmed ClosedCases share this
case's structural keys (device profile). Both move the decision: similarity to
confirmed-fraud raises p, similarity to cleared cases lowers it.

Admission policy: only bank-confirmed outcomes are citable (confirmed_fraud /
cleared from closed_cases_history.csv). Agent-resolved cases would be
provisional at half-weight until validated (policy.yaml: memory.admission) -
the graded `similar_prior_cases` field is never contaminated by them.

The same maps built here at fit time are persisted alongside the calibrator so
runtime feature vectors use the identical memory the model was trained with.
"""
from __future__ import annotations

from typing import Dict, Tuple

from bench.dataset import DatasetIndex


def device_memory_maps(idx: DatasetIndex, before: str | None = None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """device profile -> (#confirmed-fraud cases, #cleared cases)."""
    dev_fraud: Dict[str, int] = {}
    dev_cleared: Dict[str, int] = {}
    for cc in idx.closed_case_ids:
        if before is not None:
            closed_at = idx.closed_case_closed_at.get(cc)
            if not closed_at or closed_at >= before:
                continue
        devices = {idx.txn_device.get(t, "") for t in idx.closed_case_txns.get(cc, ())}
        devices.discard("")
        target = dev_fraud if cc in idx.closed_case_outcome_fraud else dev_cleared
        for d in devices:
            target[d] = target.get(d, 0) + 1
    return dev_fraud, dev_cleared


def memory_features(dev_fraud: Dict[str, int], dev_cleared: Dict[str, int],
                    devices: set) -> Tuple[int, int]:
    """Signed memory counts for this case's device set."""
    return (sum(dev_fraud.get(d, 0) for d in devices),
            sum(dev_cleared.get(d, 0) for d in devices))


def shuffled_maps(dev_fraud, dev_cleared, seed: int = 42):
    """Negative control: the statistically correct permutation null.

    Preserves each device's TOTAL case activity (T = F + C) and permutes only
    the outcome split (fraud share) across devices. Volume stays identical;
    the device->direction association is destroyed. Any decision change under
    this null is pure chance, not memory signal.
    """
    import random
    rng = random.Random(seed)
    keys = sorted(set(dev_fraud) | set(dev_cleared))
    totals = {d: dev_fraud.get(d, 0) + dev_cleared.get(d, 0) for d in keys}
    shares = sorted(d for d in keys if totals[d] > 0)
    fracs = [dev_fraud.get(d, 0) / totals[d] for d in shares]
    rng.shuffle(fracs)
    new_f, new_c = {}, {}
    for d, frac in zip(shares, fracs):
        f = round(frac * totals[d])
        new_f[d] = f
        new_c[d] = totals[d] - f
    return new_f, new_c
