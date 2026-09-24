#!/usr/bin/env python3
"""Census of normalize() rejections across ALL scaled traces.

Prints the distribution of rejection reasons so the mapping bug is unambiguous,
split by the bank's ground-truth outcome (the fraud path fails far more often).
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/mnt/sih26-train/sumora")
from normalize_traces import normalize  # noqa: E402

RAW = Path("/mnt/sih26-train/sumora/traces/traces_scaled_raw.jsonl")

recs = [json.loads(l) for l in RAW.read_text().splitlines()]
ids = sorted({r.get("case_id", "") for r in recs if r.get("case_id")})
id_map = {cid: f"HHG-{100 + (i % 900):03d}" for i, cid in enumerate(ids)}

reasons = Counter()
by_outcome = Counter()
kept = Counter()
for r in recs:
    old = r.get("case_id", "")
    new = id_map.get(old, old)
    rec = dict(r)
    rec["case_id"] = new
    rec["messages"] = [{"role": m["role"], "content": m["content"].replace(old, new)}
                       for m in r["messages"]]
    norm, why = normalize(rec)
    if norm is None:
        key = (why or "unknown")[:80]
        reasons[key] += 1
        by_outcome[(r.get("outcome"), key)] += 1
    else:
        expected = "fraud" if r.get("outcome") == "confirmed_fraud" else "legitimate"
        if norm["case"]["verdict"] == expected:
            kept[r.get("outcome")] += 1
        else:
            reasons[f"VERDICT_GATE {r.get('outcome')} -> {norm['case']['verdict']}"] += 1

print("=== top rejection reasons ===")
for why, n in reasons.most_common(10):
    print(f"{n:5d}  {why}")
print("\n=== failures by ground-truth outcome ===")
per_out = Counter()
for (outcome, why), n in by_outcome.items():
    per_out[outcome] += n
print(dict(per_out))
print("\n=== kept (pass regormalize + verdict gate) ===")
print(dict(kept))
print("\n=== raw totals ===")
print(dict(Counter(r.get("outcome") for r in recs)))
