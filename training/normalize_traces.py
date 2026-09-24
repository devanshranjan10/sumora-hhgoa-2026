#!/usr/bin/env python3
"""Normalize teacher trace completions to contract-exact answer JSON (§6a).

Root-cause fix for the failed eval gate (5/20): the teacher completions taught
an ad-hoc schema (`stopping_reason`, `verdict: "investigate"`, `actions_taken`)
that violates the bench contract. The SFT target must be the deployment format
itself, so every completion is rewritten as a schema-valid answer file:

  1. extract the last balanced {...} block from the teacher completion
  2. map fields onto the §6a contract (verdict enum, stop_reason,
     next_best_actions with policy-correct routes, evidence items, sar)
  3. fill required fields with schema-valid defaults
  4. validate against bench/schema.ANSWER_SCHEMA; drop any that still fail

Writes traces_norm.jsonl: prompt unchanged, completion = contract JSON.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/sih26-train/sumora")
SRC = BASE / "traces" / "traces_dedup.jsonl"
DST = BASE / "traces" / "traces_norm.jsonl"
sys.path.insert(0, str(BASE))  # bench/schema.py is shipped next to this script
from bench.schema import ANSWER_SCHEMA, ACTIONS  # noqa: E402

import jsonschema  # noqa: E402

VERDICT_MAP = {
    "fraud": "fraud",
    "legitimate": "legitimate",
    "cleared": "legitimate",
    "no_fraud": "legitimate",
    "uncertain": "uncertain",
    "investigate": "uncertain",
    "needs_investigation": "uncertain",
}
SOURCE_MAP = {
    "transaction": "graph", "transactions": "graph", "graph": "graph",
    "graph_edges": "graph", "edge": "graph", "device": "graph",
    "email": "graph", "cluster": "graph", "link_analysis": "graph",
    "historical_behavior": "graph", "identity_features": "graph",
    "customer": "customer", "customer_validation": "customer",
    "document": "document", "external": "external",
}
AUTO = {"ALLOW_TRANSACTION", "MONITOR_CARD", "MONITOR_CONNECTED_CARDS",
        "WARN_CUSTOMER", "VERIFY_WITH_CUSTOMER", "STEP_UP_AUTH",
        "GENERATE_REPORT", "CREATE_CASE", "ESCALATE_TO_ANALYST", "CLOSE_NO_FRAUD"}
L2_ALWAYS = {"BLOCK_ALL_CARDS", "FILE_REPORT"}


def _balanced_blocks(text: str):
    """Top-level balanced {...} spans, in a single O(n) pass.

    Scanning backwards for the last '{' (the v1 approach) grabs the INNERMOST
    trailing object — for a nested answer that is the `sar` sub-object, not the
    answer. This returns every depth-0 block instead.
    """
    blocks = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    blocks.append((start, i + 1))
                    start = -1
    return blocks


def extract_last_json(text: str):
    """The answer-shaped JSON object in a teacher completion.

    Prefers the largest top-level block carrying a `case_id` (an answer), and
    falls back to the largest parseable dict. Robust to nested `sar` /
    `next_best_actions` objects and to inline objects in the reasoning prose.
    """
    best = None
    best_len = -1
    for start, end in _balanced_blocks(text):
        try:
            obj = json.loads(text[start:end])
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if "case_id" in obj:
            return obj  # answer-shaped wins outright
        if end - start > best_len:
            best, best_len = obj, end - start
    return best


def policy_route(action: str, exposure: float) -> str:
    if action in AUTO:
        return "auto"
    if action == "DECLINE_TRANSACTION":
        return "L1"
    if action == "BLOCK_CARD":
        return "L2" if exposure > 2500.0 else "L1"
    if action in L2_ALWAYS:
        return "L2"
    return "auto"


def as_action_items(raw, exposure: float):
    items = []
    if not isinstance(raw, list):
        return items
    for a in raw:
        if isinstance(a, str):
            act = a.strip().upper()
            if act in ACTIONS:
                items.append({"action": act, "route": policy_route(act, exposure),
                              "reason": f"{act} per policy"})
        elif isinstance(a, dict):
            act = str(a.get("action", "")).strip().upper()
            if act in ACTIONS:
                items.append({"action": act,
                              "route": policy_route(act, exposure),
                              "reason": str(a.get("reason") or f"{act} per policy")})
    return items


def as_evidence(raw):
    out = []
    if isinstance(raw, list):
        for e in raw:
            if isinstance(e, str):
                src = SOURCE_MAP.get(e.lower(), "graph")
                out.append({"claim": f"Investigation signal: {e}",
                            "source": src, "ref": "teacher_trace", "entity_ids": []})
            elif isinstance(e, dict) and e.get("claim"):
                src = SOURCE_MAP.get(str(e.get("source", "")).lower(), "graph")
                out.append({"claim": str(e["claim"]), "source": src,
                            "ref": str(e.get("ref") or "teacher_trace"),
                            "entity_ids": [str(x) for x in e.get("entity_ids", [])]})
    return out


def normalize(rec: dict):
    ans = extract_last_json(rec["messages"][1]["content"])
    if not isinstance(ans, dict):
        return None, "no parseable JSON object"
    case_id = rec.get("case_id", "")
    if not re.fullmatch(r"HHG-\d{3}", str(case_id)):
        return None, f"bad case_id {case_id!r}"

    verdict = VERDICT_MAP.get(str(ans.get("verdict", "")).lower())
    if verdict is None:
        return None, f"unmappable verdict {ans.get('verdict')!r}"
    try:
        p = float(ans.get("fraud_probability", 0.5))
    except (TypeError, ValueError):
        return None, "bad fraud_probability"
    p = min(max(p, 0.0), 1.0)

    stop = str(ans.get("stop_reason") or ans.get("stopping_reason") or "").strip() \
        or "evidence reviewed"
    rules = [r for r in (ans.get("primary_rules") or []) if isinstance(r, str)]
    if rules:
        stop = f"{stop} (rules fired: {', '.join(rules)})"

    exposure = ans.get("exposure_usd", 0.0)
    try:
        exposure = max(0.0, float(exposure))
    except (TypeError, ValueError):
        exposure = 0.0

    final_actions = as_action_items(ans.get("actions_taken")
                                    or ans.get("next_best_actions", {}).get("final")
                                    if isinstance(ans.get("next_best_actions"), dict)
                                    else ans.get("actions_taken"), exposure)
    if not final_actions:
        # contract requires >=1 action; derive a policy-consistent default
        if verdict == "fraud":
            final_actions = [{"action": "BLOCK_CARD",
                              "route": policy_route("BLOCK_CARD", exposure),
                              "reason": "confirmed fraud; block card"}]
        elif verdict == "legitimate":
            final_actions = [{"action": "CLOSE_NO_FRAUD", "route": "auto",
                              "reason": "no fraud evidence"}]
        else:
            final_actions = [{"action": "VERIFY_WITH_CUSTOMER", "route": "auto",
                              "reason": "insufficient evidence; verify"}]

    if verdict == "legitimate":
        affected, exposure = [], 0.0
    else:
        affected = [str(x) for x in (ans.get("affected_txn_ids") or [])]
    pattern = ans.get("pattern")
    pattern = pattern if pattern in ("card_testing", "card_not_present_fraud",
                                     "card_not_present_new_device", "out_of_region_use",
                                     "account_takeover", "undocumented", "none") else "none"
    pattern_desc = str(ans.get("pattern_description") or "") if pattern == "undocumented" else ""

    sar_raw = ans.get("sar") if isinstance(ans.get("sar"), dict) else {}
    sar_file = bool(sar_raw.get("file")) or any(
        a["action"] == "FILE_REPORT" for a in final_actions)
    sar_reason = str(sar_raw.get("reason") or "").strip() or (
        f"activity consistent with {pattern or 'reported fraud'}: {stop}" if sar_file
        else "no SAR: criteria not met")
    dates = [d for d in (sar_raw.get("activity_dates") or [])
             if isinstance(d, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)][:2]

    evidence_requests = []
    for er in (ans.get("evidence_requests") or []):
        if isinstance(er, dict) and er.get("type") in (
                "customer_validation", "step_up_auth", "analyst_info"):
            try:
                evidence_requests.append({
                    "type": er["type"],
                    "asked_after_step": int(er.get("asked_after_step", 0)),
                    "assumed_response": str(er.get("assumed_response", "")) or "no response",
                })
            except (TypeError, ValueError):
                pass

    status = {"fraud": "closed_fraud", "legitimate": "closed_legitimate",
              "uncertain": "escalated"}[verdict]

    norm = {
        "case_id": case_id,
        "case": {
            "status": status,
            "verdict": verdict,
            "fraud_probability": p,
            "pattern": pattern,
            "pattern_description": pattern_desc,
            "affected_txn_ids": affected,
            "first_suspicious_txn_id": str(ans.get("first_suspicious_txn_id") or ""),
            "connected_card_ids": [str(x) for x in (ans.get("connected_card_ids") or [])],
            "connected_device_profiles": [str(x) for x in (ans.get("connected_device_profiles") or [])],
            "exposure_usd": exposure,
            "evidence": as_evidence(ans.get("evidence") or ans.get("evidence_classes")),
            "similar_prior_cases": [str(x) for x in (ans.get("similar_prior_cases") or [])],
            "summary": str(ans.get("summary") or stop),
            "written_to_graph": False,
            "graph_case_id": "",
        },
        "evidence_requests": evidence_requests,
        "next_best_actions": {
            "initial": final_actions,
            "final": final_actions,
            "what_changed": str(ans.get("what_changed") or
                                "none: decision unchanged after evidence review"),
        },
        "sar": {
            "file": sar_file,
            "reason": sar_reason,
            "narrative": str(sar_raw.get("narrative") or (stop if sar_file else "")),
            "subjects": [str(x) for x in (sar_raw.get("subjects") or [])],
            "total_amount_usd": max(0.0, float(sar_raw.get("total_amount_usd") or exposure or 0.0)),
            "activity_dates": dates,
        },
        "stop_reason": stop,
        "tool_calls": len(rec.get("tool_trace") or []),
        "tokens": len(rec["messages"][1]["content"]) // 4,
        "latency_s": 0.0,
    }
    try:
        jsonschema.validate(norm, ANSWER_SCHEMA)
    except jsonschema.ValidationError as e:
        return None, f"schema: {e.message[:120]}"
    return norm, None


def main() -> None:
    n_in = n_ok = 0
    reasons: dict = {}
    out_lines = []
    for line in SRC.read_text().splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        n_in += 1
        norm, why = normalize(rec)
        if norm is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        out_lines.append(json.dumps({
            "case_id": rec["case_id"], "model": rec.get("model"), "trial": rec.get("trial"),
            "messages": [{"role": "user", "content": rec["messages"][0]["content"]},
                         {"role": "assistant", "content": json.dumps(norm, indent=2)}],
        }))
        n_ok += 1
    DST.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    print(f"input={n_in} normalized={n_ok} dropped={n_in - n_ok}")
    print("drop reasons:", json.dumps(reasons, indent=2))


if __name__ == "__main__":
    main()
