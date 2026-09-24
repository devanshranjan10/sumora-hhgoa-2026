#!/usr/bin/env python3
"""Normalize + ground-truth-gate the scaled teacher traces (SOTA curriculum).

Input : traces/traces_scaled_raw.jsonl  (from gen_traces_scaled.py; carries the
        bank's ground-truth `outcome` + `pattern` per record)
Output: traces/traces_scaled_norm.jsonl (contract-exact §6a completions)

Three quality gates the v1 54-trace run lacked:

  1. Ground-truth verdict gate — a trace only teaches when the teacher's verdict
     agrees with the bank's closed-case outcome (confirmed_fraud -> fraud,
     cleared -> legitimate). Disagreements are DROPPED and counted; they would
     teach the model to contradict the answer key.
  2. Pattern authority — the bank's pattern label wins (the grader's key derives
     from it). Teacher pattern disagreements are overwritten, not trained on.
  3. p anchoring — confirmed-fraud targets are clamped to [0.85, 0.97] and
     cleared targets to [0.03, 0.15], so the model learns the calibrated bands
     the README stopping rule expects instead of a regressed-to-mean 0.5.

Run on the A100:  ./venv/bin/python normalize_scaled.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

BASE = Path("/mnt/sih26-train/sumora")
SRC = BASE / "traces" / "traces_scaled_raw.jsonl"
DST = BASE / "traces" / "traces_scaled_norm.jsonl"
sys.path.insert(0, str(BASE))
from bench.schema import ANSWER_SCHEMA  # noqa: E402
from normalize_traces import normalize  # noqa: E402  (reuses the §6a mapping)

import jsonschema  # noqa: E402

rng = random.Random(42)


def anchor_p(p: float, outcome: str) -> float:
    """Clamp the teacher's p into the calibrated band for the bank's outcome."""
    if outcome == "confirmed_fraud":
        lo, hi = 0.85, 0.97
    else:
        lo, hi = 0.03, 0.15
    return round(min(hi, max(lo, p)), 2)


def remap_case_id(rec: dict, mapping: dict) -> dict:
    """Closed cases use CC-xxxx ids; the answer contract requires HHG-###.

    The id is remapped in BOTH the prompt and the completion so prompt and
    target agree (the model learns to echo the id it was given, which is exactly
    what the 20 exam cases — HHG-001..HHG-020 — need).
    """
    old = rec.get("case_id", "")
    new = mapping.get(old)
    if not new:
        return rec
    out = dict(rec)
    out["case_id"] = new
    out["messages"] = [
        {"role": m["role"], "content": m["content"].replace(old, new)}
        for m in rec["messages"]
    ]
    return out


def build_id_map(raw_records: list) -> dict:
    ids = sorted({r.get("case_id", "") for r in raw_records if r.get("case_id")})
    return {cid: f"HHG-{100 + (i % 900):03d}" for i, cid in enumerate(ids)}


def main() -> None:
    raw = []
    for line in SRC.read_text().splitlines():
        try:
            raw.append(json.loads(line))
        except Exception:
            continue
    id_map = build_id_map(raw)
    print(f"id map built for {len(id_map)} closed cases", flush=True)

    n_in = n_kept = n_verdict_drop = n_schema_drop = 0
    out_lines = []
    verdict_counts: dict = {}
    for raw_rec in raw:
        rec = remap_case_id(raw_rec, id_map)
        n_in += 1
        norm, why = normalize(rec)
        if norm is None:
            n_schema_drop += 1
            continue
        outcome = rec.get("outcome", "")
        expected = "fraud" if outcome == "confirmed_fraud" else "legitimate"
        verdict = norm["case"]["verdict"]
        if verdict != expected:
            n_verdict_drop += 1
            continue
        # Bank's pattern label is authoritative (the answer key derives from it).
        bank_pattern = rec.get("pattern") or "none"
        norm["case"]["pattern"] = bank_pattern
        norm["case"]["pattern_description"] = (
            norm["case"].get("pattern_description", "") if bank_pattern == "undocumented" else ""
        )
        # Calibrated p band per outcome.
        try:
            p = float(norm["case"]["fraud_probability"])
        except (TypeError, ValueError):
            p = 0.9 if expected == "fraud" else 0.08
        norm["case"]["fraud_probability"] = anchor_p(p, outcome)
        try:
            jsonschema.validate(norm, ANSWER_SCHEMA)
        except jsonschema.ValidationError:
            n_schema_drop += 1
            continue
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        out_lines.append(json.dumps({
            "case_id": rec["case_id"], "trial": rec.get("trial", 0),
            "outcome": outcome, "pattern": bank_pattern,
            "messages": [{"role": "user", "content": rec["messages"][0]["content"]},
                         {"role": "assistant", "content": json.dumps(norm, indent=2)}],
        }))
        n_kept += 1

    rng.shuffle(out_lines)
    DST.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    fraud_n = sum(1 for l in out_lines if '"confirmed_fraud"' in l)
    print(f"input={n_in} kept={n_kept} verdict_drops={n_verdict_drop} "
          f"schema_drops={n_schema_drop}")
    print(f"class balance: fraud={fraud_n} cleared={n_kept - fraud_n}")
    print("verdicts:", json.dumps(verdict_counts))
    print("NORMALIZE_SCALED_DONE", flush=True)


if __name__ == "__main__":
    main()