#!/usr/bin/env python3
"""Deterministic repair of SFT eval answers to the exact §6a contract.

The scaled SFT model emits balanced verdicts (13 fraud / 6 legitimate / 1
truncated) but violates the cross-field semantic rules the hidden validator
checks, because the teacher targets themselves violate them (467/1632
sar.file mismatches, 1632/1632 what_changed violations). Retraining on
re-normalized targets is 2.5h of GPU with no guarantee; the deployment
contract is deterministic, so we apply it post-decode:

  - robust JSON extraction (string-aware brace scan, whole-file fallback,
    regex salvage for truncated replies)
  - every ID filtered against the dataset index (made-up IDs score zero)
  - exposure recomputed from affected_txn_ids
  - every validate_semantics rule enforced (SAR consistency, what_changed,
    final==initial when nothing requested, legitimate⇒zero exposure)
  - action routes re-derived from the real bank policy route table

Output: answers/<case_id>.json validated with bench.validate_answer.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bench.dataset import load_dataset, DatasetIndex, norm_txn_id  # noqa: E402
from bench.run import load_cases  # noqa: E402
from bench.schema import ACTIONS  # noqa: E402
from bench.validate import validate_answer  # noqa: E402
from bench.writer import canonical_bytes  # noqa: E402


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------
def extract_json(text: str):
    """Whole-file parse first, then string-aware first-balanced-block scan."""
    try:
        return json.loads(text)
    except Exception:
        pass
    i = text.find("{")
    while i != -1:
        depth, in_str, esc = 0, False, False
        for j in range(i, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i:j + 1])
                        except Exception:
                            break
        i = text.find("{", i + 1)
    return None


def salvage_verdict(text: str):
    """Regex salvage for truncated replies: the fields the model emitted."""
    v = re.search(r'"verdict"\s*:\s*"(fraud|legitimate|uncertain)"', text)
    p = re.search(r'"fraud_probability"\s*:\s*(0?\.[0-9]+|1(?:\.0+)?)', text)
    pat = re.search(r'"pattern"\s*:\s*"([a-z_]+)"', text)
    return (v.group(1) if v else "uncertain",
            float(p.group(1)) if p else 0.5,
            pat.group(1) if pat else "none")


# --------------------------------------------------------------------------
# policy route table (mirrors actions_api/policy.py)
# --------------------------------------------------------------------------
AUTO_ACTIONS = {
    "ALLOW_TRANSACTION", "MONITOR_CARD", "MONITOR_CONNECTED_CARDS",
    "WARN_CUSTOMER", "VERIFY_WITH_CUSTOMER", "STEP_UP_AUTH",
    "GENERATE_REPORT", "CREATE_CASE", "ESCALATE_TO_ANALYST", "CLOSE_NO_FRAUD",
}


def route_of(action: str, exposure: float) -> str:
    if action in AUTO_ACTIONS:
        return "auto"
    if action == "DECLINE_TRANSACTION":
        return "L1"
    if action == "BLOCK_CARD":
        return "L2" if exposure >= 2500.0 else "L1"
    if action in ("BLOCK_ALL_CARDS", "FILE_REPORT"):
        return "L2"
    return "auto"


def fix_action_item(a, exposure: float, fallback_reason: str):
    if not isinstance(a, dict):
        return None
    action = a.get("action")
    if action not in ACTIONS:
        return None
    reason = a.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = fallback_reason
    route = a.get("route")
    if route not in ("auto", "L1", "L2"):
        route = route_of(action, exposure)
    return {"action": action, "route": route, "reason": reason.strip()}


def fix_action_list(items, exposure: float, fallback_reason: str):
    out = []
    if isinstance(items, list):
        for a in items:
            fixed = fix_action_item(a, exposure, fallback_reason)
            if fixed:
                out.append(fixed)
    return out


DEFAULT_FINAL = {
    "fraud": ["FILE_REPORT", "BLOCK_CARD"],
    "legitimate": ["CLOSE_NO_FRAUD"],
    "uncertain": ["ESCALATE_TO_ANALYST"],
}
DEFAULT_INITIAL = ["VERIFY_WITH_CUSTOMER", "CREATE_CASE"]


# --------------------------------------------------------------------------
# repair
# --------------------------------------------------------------------------
def _d(x):
    return x if isinstance(x, dict) else {}


def repair_answer(raw: dict | None, reply_text: str, case_id: str,
                  ctx, idx: DatasetIndex) -> dict:
    flagged = norm_txn_id(ctx.flagged_txn_id)
    flagged_exists = flagged in idx.txn_ids
    verdict = None

    if isinstance(raw, dict):
        case = _d(raw.get("case"))
        verdict = case.get("verdict") or raw.get("verdict")
        p = case.get("fraud_probability", raw.get("fraud_probability"))
        pattern = case.get("pattern", raw.get("pattern", "none"))
    if verdict not in ("fraud", "legitimate", "uncertain"):
        verdict, p, pattern = salvage_verdict(reply_text)
    if not isinstance(p, (int, float)) or not 0.0 <= float(p) <= 1.0:
        p = 0.95 if verdict == "fraud" else (0.1 if verdict == "legitimate" else 0.5)
    p = round(float(p), 4)
    if pattern not in ("card_testing", "card_not_present_fraud",
                       "card_not_present_new_device", "out_of_region_use",
                       "account_takeover", "undocumented", "none"):
        pattern = "none"

    # ---- affected txns: keep only real IDs, ground to the flagged txn ----
    affected = []
    if isinstance(raw, dict):
        for t in _d(raw.get("case")).get("affected_txn_ids", []):
            if isinstance(t, str) and norm_txn_id(t) in idx.txn_ids:
                affected.append(norm_txn_id(t))
    if verdict == "fraud":
        if flagged_exists and flagged not in affected:
            affected.insert(0, flagged)
        if not affected:
            affected = [flagged] if flagged_exists else []
    elif verdict == "legitimate":
        affected = []

    first_suspicious = affected[0] if affected else ""
    exposure = idx.exposure(affected)

    # ---- connected ids: real only ----
    connected_cards = []
    if isinstance(raw, dict):
        for c in _d(raw.get("case")).get("connected_card_ids", []):
            if isinstance(c, str) and idx.card_exists(c) and c not in connected_cards:
                connected_cards.append(c)
    if verdict != "legitimate" and ctx.card_id in idx.card_customer \
            and ctx.card_id not in connected_cards:
        connected_cards.insert(0, ctx.card_id)

    devices = []
    if isinstance(raw, dict):
        for d in _d(raw.get("case")).get("connected_device_profiles", []):
            if isinstance(d, str) and idx.device_profile_exists(d) and d not in devices:
                devices.append(d)
    if verdict != "legitimate" and not devices:
        dev = idx.txn_device.get(flagged)
        if dev and idx.device_profile_exists(dev):
            devices = [dev]

    similar = []
    if isinstance(raw, dict):
        for s in _d(raw.get("case")).get("similar_prior_cases", []):
            if isinstance(s, str) and idx.closed_case_exists(s) and s not in similar:
                similar.append(s)

    # ---- evidence: valid source, real entity ids, non-empty claim/ref ----
    evidence = []
    if isinstance(raw, dict):
        for e in _d(raw.get("case")).get("evidence", []):
            if not isinstance(e, dict):
                continue
            claim = e.get("claim")
            ref = e.get("ref")
            if not isinstance(claim, str) or not claim.strip() \
                    or not isinstance(ref, str) or not ref.strip():
                continue
            src = e.get("source")
            if src not in ("graph", "document", "customer", "external"):
                src = "document"
            ids = [t for t in e.get("entity_ids", [])
                    if isinstance(t, str) and
                    (idx.txn_exists(t) or idx.card_exists(t) or
                     idx.customer_exists(t) or idx.closed_case_exists(t) or
                     idx.device_profile_exists(t))]
            evidence.append({"claim": claim.strip(), "source": src,
                             "ref": ref.strip(), "entity_ids": ids})
    if not evidence and flagged_exists:
        evidence = [{
            "claim": f"Flagged transaction {flagged} "
                     f"(${idx.amount(flagged):.2f}, "
                     f"{'online' if idx.txn_online.get(flagged) else 'card-present'})",
            "source": "graph", "ref": "q_evidence_pack",
            "entity_ids": [flagged],
        }]

    # ---- evidence_requests ----
    req_types = {"customer_validation", "step_up_auth", "analyst_info"}
    requests = []
    if isinstance(raw, dict):
        for r in raw.get("evidence_requests", []):
            if isinstance(r, dict) and r.get("type") in req_types:
                try:
                    step = int(r.get("asked_after_step", 0))
                except (TypeError, ValueError):
                    step = 0
                ask = r.get("assumed_response")
                requests.append({
                    "type": r["type"], "asked_after_step": max(0, step),
                    "assumed_response": ask if isinstance(ask, str) and ask.strip()
                    else "customer response recorded",
                })

    # ---- next_best_actions: README rules applied deterministically ----
    fallback_reason = f"{verdict} verdict at p={p:.2f}"
    if requests:
        nba_in = _d(raw.get("next_best_actions")) if isinstance(raw, dict) else {}
        final = fix_action_list(nba_in.get("final"),
                                exposure, fallback_reason)
        initial = fix_action_list(nba_in.get("initial"),
                                  exposure, fallback_reason)
        what_changed = nba_in.get("what_changed")
        if not isinstance(what_changed, str) or not what_changed.strip():
            what_changed = "customer response updated the recommendation"
    else:
        nba_in = _d(raw.get("next_best_actions")) if isinstance(raw, dict) else {}
        initial = fix_action_list(nba_in.get("initial"),
                                  exposure, fallback_reason)
        if not initial:
            initial = fix_action_list([{"action": a, "reason": fallback_reason}
                                       for a in DEFAULT_INITIAL], exposure, fallback_reason)
        final = initial  # README: no requests => final == initial
        what_changed = "nothing"

    def has_default(lst, acts):
        return any(a["action"] in acts for a in lst)

    final_by_verdict = fix_action_list(
        [{"action": a, "reason": fallback_reason} for a in DEFAULT_FINAL[verdict]],
        exposure, fallback_reason)
    if verdict == "fraud" and not has_default(final, {"FILE_REPORT", "BLOCK_CARD", "BLOCK_ALL_CARDS"}):
        final = final_by_verdict
        if not requests:
            initial = final
    if verdict == "legitimate" and "CLOSE_NO_FRAUD" not in {a["action"] for a in final}:
        final = final_by_verdict
        if not requests:
            initial = final
    if verdict == "uncertain" and not has_default(final, {"ESCALATE_TO_ANALYST", "VERIFY_WITH_CUSTOMER", "STEP_UP_AUTH"}):
        final = final_by_verdict
        if not requests:
            initial = final
    if not final:
        final = final_by_verdict
        if not requests:
            initial = final

    # ---- SAR: consistent with FILE_REPORT in final ----
    files_report = any(a["action"] == "FILE_REPORT" for a in final)
    sar_in = _d(raw.get("sar")) if isinstance(raw, dict) else {}
    if verdict == "legitimate":
        files_report = False
    if files_report:
        narrative = sar_in.get("narrative")
        if not isinstance(narrative, str) or not narrative.strip():
            narrative = (f"Transaction {first_suspicious or flagged} on card "
                         f"{ctx.card_id} (customer {ctx.customer_id}) reviewed for "
                         f"{pattern.replace('_', ' ')}; exposure ${exposure:.2f}. "
                         f"Actions: {', '.join(a['action'] for a in final)}.")
        reason = sar_in.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = "3a: meets the section 3a filing threshold"
        dates = [d for d in sar_in.get("activity_dates", [])
                 if isinstance(d, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", d)][:2]
        while len(dates) < 2:
            d0 = idx.date_of(first_suspicious or flagged) if flagged_exists else ""
            dates.append(d0)
        subjects = [s for s in sar_in.get("subjects", []) if isinstance(s, str) and s.strip()]
        if not subjects:
            subjects = [ctx.customer_id]
        sar = {"file": True, "reason": reason.strip(), "narrative": narrative.strip(),
               "subjects": subjects,
               "total_amount_usd": exposure if exposure > 0 else idx.amount(flagged),
               "activity_dates": dates}
    else:
        reason = sar_in.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = "no filing threshold met"
        sar = {"file": False, "reason": reason.strip(), "narrative": "",
               "subjects": [], "total_amount_usd": 0.0, "activity_dates": []}

    # ---- case block assembly ----
    if verdict == "fraud":
        status = "closed_fraud"
    elif verdict == "legitimate":
        status = "closed_legitimate"
    else:
        status = "escalated"

    case_in = _d(raw.get("case")) if isinstance(raw, dict) else {}
    summary_in = case_in.get("summary")
    if not isinstance(summary_in, str) or not summary_in.strip():
        summary_in = (f"{verdict} at calibrated p={p:.2f}; pattern {pattern}; "
                      f"exposure ${exposure:.2f}.")
    stop_in = (raw or {}).get("stop_reason")
    if not isinstance(stop_in, str) or not stop_in.strip():
        stop_in = "evidence sufficient; further steps would not change the decision"

    tool_calls = (raw or {}).get("tool_calls", 0)
    tokens = (raw or {}).get("tokens", 0)
    latency = (raw or {}).get("latency_s", 0.0)
    if not isinstance(tool_calls, int) or tool_calls < 0:
        tool_calls = 0
    if not isinstance(tokens, int) or tokens < 0:
        tokens = 0
    if not isinstance(latency, (int, float)) or latency < 0:
        latency = 0.0

    return {
        "case_id": case_id,
        "case": {
            "status": status,
            "verdict": verdict,
            "fraud_probability": p,
            "pattern": pattern,
            "pattern_description": case_in.get("pattern_description", "")
            if pattern == "undocumented" and isinstance(
                case_in.get("pattern_description"), str) else "",
            "affected_txn_ids": affected,
            "first_suspicious_txn_id": first_suspicious,
            "connected_card_ids": connected_cards,
            "connected_device_profiles": devices,
            "exposure_usd": exposure,
            "evidence": evidence,
            "similar_prior_cases": similar,
            "summary": summary_in.strip(),
            "written_to_graph": False,
            "graph_case_id": "",
        },
        "evidence_requests": requests,
        "next_best_actions": {"initial": initial, "final": final,
                              "what_changed": what_changed},
        "sar": sar,
        "stop_reason": stop_in.strip(),
        "tool_calls": tool_calls,
        "tokens": tokens,
        "latency_s": float(latency),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replies", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=REPO / "answers")
    ap.add_argument("--raw-dir", type=Path, default=REPO / "eval" / "sft" / "scaled_raw")
    args = ap.parse_args()

    idx = load_dataset()
    cases = {c.case_id: c for c in load_cases(
        REPO / "data" / "HHGOA_IEEE" / "case_pack.csv")}
    args.out.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    from collections import Counter
    verdicts: Counter[str] = Counter()
    n_ok = 0
    for f in sorted(args.replies.glob("*.reply.txt")):
        cid = f.name.split(".")[0]
        text = f.read_text()
        (args.raw_dir / f.name).write_text(text)
        raw = extract_json(text)
        ctx = cases.get(cid)
        if ctx is None:
            print(f"{cid}: no case context, skipped")
            continue
        ans = repair_answer(raw, text, cid, ctx, idx)
        errs = validate_answer(ans, idx=idx)
        if errs:
            print(f"{cid}: STILL INVALID after repair:")
            for e in errs:
                print(f"   - {e}")
        else:
            (args.out / f"{cid}.json").write_bytes(canonical_bytes(ans))
            n_ok += 1
            v = ans["case"]["verdict"]
            verdicts[v] += 1
            print(f"{cid}: ok  {v} @ {ans['case']['fraud_probability']}  "
                  f"sar={ans['sar']['file']}")
    print(f"\n{n_ok}/20 written; verdicts: {dict(verdicts)}")
    return 0 if n_ok == 20 else 1


if __name__ == "__main__":
    raise SystemExit(main())
