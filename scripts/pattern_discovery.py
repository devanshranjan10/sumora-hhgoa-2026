#!/usr/bin/env python3
"""Undocumented-pattern discovery with permutation negative control.

Replaces the weak pack-vs-closed scalar enrichment test. The headline claim
(spec §5 / winning strategy §4):

  Build-specific device profiles that link ≥3 distinct cardholders on
  bank-confirmed fraud cases form a structural typology the five documented
  patterns do not cover ("cross-account shared-device ring").

Negative control: shuffle confirmed-fraud labels across closed cases, recount
rings. Only ship if p < 0.05. The LLM never names a pattern before this gate.

Offline (CSV index) — no TigerGraph required. Live mode can re-run the same
statistic over Louvain communities via gsql/discovery.gsql when TG is up.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.dataset import load_dataset  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "HHGOA_IEEE"
OUT = REPO / "eval" / "pattern_discovery.json"
N_PERM = 2000
SEED = 20260920
MIN_CUSTOMERS = 3
P_THRESH = 0.05

# Typology we will claim if the gate passes (R9: describe, don't force a
# documented label). Name is stable for answer files + UI.
TYPOLOGY_ID = "cross_account_device_ring"
TYPOLOGY_NAME = "Cross-account shared-device ring"
TYPOLOGY_DESCRIPTION = (
    "A single Build/-qualified device profile appears as a new device on "
    "confirmed-fraud cases spanning ≥3 distinct cardholders. Analysts already "
    "flagged the SM-G935F Chrome/Android exemplar as unmatched to the five "
    "documented typologies (ATO / CNP / card-testing / out-of-region / "
    "CNP-new-device). The ring is structural: shared fingerprint across "
    "accounts, not a single-account behavioral burst."
)


def _load_case_meta() -> Dict[str, dict]:
    with (DATA / "closed_cases_history.csv").open(newline="") as fh:
        return {r["case_id"]: r for r in csv.DictReader(fh)}


def _build_device_index(idx, meta) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """device -> closed-case ids; device -> customer ids (via case rows)."""
    dev_cases: Dict[str, Set[str]] = defaultdict(set)
    for ccid, txns in idx.closed_case_txns.items():
        for t in txns:
            d = idx.txn_device.get(t, "")
            if not d or "Build/" not in d:
                continue
            dev_cases[d].add(ccid)
    return dev_cases, meta


def count_rings(
    dev_cases: Dict[str, Set[str]],
    meta: Dict[str, dict],
    fraud_set: Set[str],
    min_customers: int = MIN_CUSTOMERS,
) -> Tuple[int, List[dict]]:
    """Build/-specific devices linking ≥min_customers distinct fraud cardholders."""
    exemplars: List[dict] = []
    for device, cset in dev_cases.items():
        fraud_cs = [c for c in cset if c in fraud_set]
        if len(fraud_cs) < min_customers:
            continue
        custs = {meta.get(c, {}).get("customer_id", "") for c in fraud_cs}
        custs.discard("")
        if len(custs) < min_customers:
            continue
        exemplars.append({
            "device": device,
            "n_customers": len(custs),
            "n_fraud_cases": len(fraud_cs),
            "case_ids": sorted(fraud_cs),
            "customer_ids": sorted(custs),
            "patterns": sorted({
                meta.get(c, {}).get("pattern", "") for c in fraud_cs
            }),
        })
    exemplars.sort(key=lambda e: (-e["n_customers"], -e["n_fraud_cases"], e["device"]))
    return len(exemplars), exemplars


def permutation_p(
    dev_cases: Dict[str, Set[str]],
    meta: Dict[str, dict],
    all_cc: List[str],
    fraud_set: Set[str],
    observed: int,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> Tuple[float, List[int]]:
    rng = random.Random(seed)
    flags = [c in fraud_set for c in all_cc]
    nulls: List[int] = []
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(flags)
        fs = {c for c, f in zip(all_cc, flags) if f}
        cnt, _ = count_rings(dev_cases, meta, fs)
        nulls.append(cnt)
        if cnt >= observed:
            ge += 1
    return (ge + 1) / (n_perm + 1), nulls


def main() -> None:
    idx = load_dataset()
    meta = _load_case_meta()
    dev_cases, _ = _build_device_index(idx, meta)
    fraud_set = set(idx.closed_case_outcome_fraud)
    all_cc = list(idx.closed_case_ids)

    observed, exemplars = count_rings(dev_cases, meta, fraud_set)
    p_value, nulls = permutation_p(dev_cases, meta, all_cc, fraud_set, observed)
    nulls_sorted = sorted(nulls)
    significant = p_value < P_THRESH

    # Prefer the SM-G935F ring that matches bank-labeled undocumented cases
    # as the demo exemplar; fall back to the densest ring.
    und_cases = {
        cid for cid, row in meta.items() if row.get("pattern") == "undocumented"
    }
    demo = None
    for e in exemplars:
        if any(c in und_cases for c in e["case_ids"]) and "SM-G935F" in e["device"]:
            demo = e
            break
    if demo is None and exemplars:
        demo = exemplars[0]

    # Distance-from-documented: rings whose fraud cases already carry mixed /
    # undocumented labels are farther from a single documented centroid.
    if demo is not None:
        demo = dict(demo)
        demo["matches_bank_undocumented"] = any(
            c in und_cases for c in demo["case_ids"]
        )

    result = {
        "method": "build_specific_cross_account_device_rings",
        "statistic": (
            f"count of Build/-qualified device profiles linking "
            f">={MIN_CUSTOMERS} distinct cardholders on confirmed-fraud cases"
        ),
        "n_closed": len(all_cc),
        "n_fraud": len(fraud_set),
        "n_perm": N_PERM,
        "seed": SEED,
        "min_customers": MIN_CUSTOMERS,
        "observed_ring_count": observed,
        "null_mean": round(sum(nulls) / len(nulls), 4) if nulls else None,
        "null_p95": nulls_sorted[int(0.95 * (len(nulls) - 1))] if nulls else None,
        "perm_p_value": round(p_value, 6),
        "significant_at_0.05": significant,
        "typology": {
            "pattern": "undocumented",
            "pattern_id": TYPOLOGY_ID,
            "name": TYPOLOGY_NAME,
            "pattern_description": TYPOLOGY_DESCRIPTION,
            "shipped": significant,
        },
        "demo_exemplar": demo,
        "top_exemplars": exemplars[:10],
        "bank_undocumented_seed_cases": sorted(und_cases),
        # Keep legacy keys so older readers don't crash; mark superseded.
        "legacy_pack_enrichment": "superseded — see method field",
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "significant_at_0.05": significant,
        "perm_p_value": result["perm_p_value"],
        "observed_ring_count": observed,
        "null_mean": result["null_mean"],
        "typology_shipped": significant,
        "demo_device": (demo or {}).get("device", "")[:80],
        "demo_cases": (demo or {}).get("case_ids", []),
        "out": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
