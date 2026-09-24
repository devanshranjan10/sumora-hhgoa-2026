#!/usr/bin/env python3
"""Sumora scaled trace generation — the SOTA curriculum fix.

Why: the v1 run trained on 54 traces from the 20 exam cases (94% fraud base
rate) and the SFT model collapsed to the modal answer. The fix is the tiered
curriculum the spec demands (section 6b), built from the bank's OWN labeled
history — 5,565 closed cases (4,665 fraud / 900 cleared) — which the README
explicitly calls "your agent's starting memory".

What this does:
  1. Stratified sample of closed cases: ALL 900 cleared + ~1,200 fraud
     stratified across the 5 documented patterns + undocumented.
  2. For each case, pull a budgeted evidence digest from the LIVE TigerGraph
     (same query set as the deployed agent: card-testing scorer, new-device
     CNP, out-of-region, k-hop neighborhood).
  3. Ask Gemini 2.5 Pro to investigate under the REAL bank policy (verbatim
     R1-R10 from the dataset README — the v1 prompt taught a fictional policy,
     which poisoned every rule citation downstream).
  4. Emit raw records {case_id, outcome, pattern, messages, tool_trace}.
     Ground-truth filtering + p-anchoring happens in normalize_scaled.py.

Run on the A100 VM host (Vertex AI via metadata-server token, no key file):
  cd /mnt/sih26-train/sumora && nohup ./venv/bin/python gen_traces_scaled.py \
      > traces/scaled_gen.log 2>&1 &
"""
from __future__ import annotations

import csv
import json
import random
import sys
import threading
import time
import urllib.parse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

PROJECT = "contral-6b0bd"
LOCATION = "us-central1"
GRAPH = "SumoraFraudGraph"
TG = f"http://localhost:9000/query/{GRAPH}"
METADATA = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
CC_CSV = Path("/mnt/sih26-train/hhgoa/data/HHGOA_IEEE/prepped/closed_cases_history.csv")
OUT = Path("/mnt/sih26-train/sumora/traces/traces_scaled_raw.jsonl")
LOG = Path("/mnt/sih26-train/sumora/traces/scaled_gen.log")

MODEL = "gemini-2.5-pro"
WORKERS = 10
MAX_FRAUD = 1200
SEED = 20260920

# The REAL bank policy, verbatim from data/HHGOA_IEEE/README.md (Fraud Policy
# section). The v1 prompt's invented R1-R10 taught the teacher (and therefore
# the SFT model) a fictional rule set; every rule citation downstream was
# poisoned. This text is authoritative.
POLICY = """BANK FRAUD POLICY (authoritative; cite rule ids exactly as given):
Approval routes: auto = agent may act alone; L1 = team lead; L2 = fraud manager.
Rules:
R1 Verify before you block on a weak signal: if the case rests on a single signal (incl. a risk score alone) and assessed fraud probability < 0.70, recommend VERIFY_WITH_CUSTOMER or STEP_UP_AUTH before any block.
R2 Customer denies the transaction: recommend BLOCK_CARD and CREATE_CASE. Add FILE_REPORT if exposure > $1,000 or the case connects to a shared device profile or another card's fraud.
R3 Customer confirms the transaction: recommend CLOSE_NO_FRAUD.
R4 No reply within 24 hours: recommend MONITOR_CARD and DECLINE_TRANSACTION for pending authorizations. Escalate if exposure > $500.
R5 Card testing: >=3 small online authorizations on one card within an hour, then a larger purchase: DECLINE_TRANSACTION and STEP_UP_AUTH; if a purchase > $100 already cleared, BLOCK_CARD.
R6 Shared origin: several cards show fraud from the same device profile / billing region / recipient email in one window: name the shared element, CREATE_CASE and FILE_REPORT, MONITOR_CONNECTED_CARDS for every card sharing it.
R7 Disputed but legitimate: the disputed charge matches the customer's own recurring pattern (same merchant, amount, monthly): CREATE_CASE, VERIFY_WITH_CUSTOMER, WARN_CUSTOMER. Do not block.
R8 Escalate when uncertain and exposed: verdict uncertain and exposure > $500, or evidence conflicts: ESCALATE_TO_ANALYST.
R9 Undocumented patterns: activity fits none of the five known patterns but shows coordinated or repeated abuse across customers: CREATE_CASE, FILE_REPORT, ESCALATE_TO_ANALYST, and describe the pattern in your own words. Never force it into a known category.
R10 Never BLOCK_ALL_CARDS unless >=2 of the customer's cards show confirmed fraud or credentials are confirmed compromised.
Routes by action: auto = ALLOW_TRANSACTION, MONITOR_CARD, MONITOR_CONNECTED_CARDS, WARN_CUSTOMER, VERIFY_WITH_CUSTOMER, STEP_UP_AUTH, GENERATE_REPORT, CREATE_CASE, ESCALATE_TO_ANALYST, CLOSE_NO_FRAUD. L1 = DECLINE_TRANSACTION, BLOCK_CARD with exposure <= $2,500. L2 = BLOCK_CARD with exposure > $2,500, BLOCK_ALL_CARDS always, FILE_REPORT always.
SAR (FILE_REPORT) when fraud is confirmed or strongly suspected AND any of: exposure > $1,000; connects to a shared device/region/another customer's fraud; coordinated or undocumented pattern.
Stop when: p >= 0.85 or p <= 0.15 supported by >=2 independent evidence pieces; or a verification response settles it; or further steps cannot change the decision.
Known patterns: card_testing, card_not_present_fraud, card_not_present_new_device, out_of_region_use, account_takeover, undocumented (describe it), none.
"""


def vertex_token(client: httpx.Client) -> str:
    r = client.get(METADATA, headers={"Metadata-Flavor": "Google"}, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]


class TokenRefresher:
    def __init__(self, client: httpx.Client, max_age_s: int = 2400) -> None:
        self.client = client
        self.max_age_s = max_age_s
        self._token = ""
        self._ts = 0.0
        self._lock = threading.Lock()

    def get(self, force: bool = False) -> str:
        with self._lock:
            if force or not self._token or (time.time() - self._ts) > self.max_age_s:
                self._token = vertex_token(self.client)
                self._ts = time.time()
            return self._token


def call_query(client: httpx.Client, name: str, params: dict) -> dict:
    r = client.post(f"{TG}/{name}", json=params, timeout=120)
    r.raise_for_status()
    return r.json()


def safe(client: httpx.Client, q: str, params: dict) -> dict:
    try:
        return call_query(client, q, params)
    except Exception as e:
        return {"error": str(e)[:120]}


def _compact(obj, depth: int = 0):
    """Flatten TG JSON envelopes into a small readable digest."""
    if isinstance(obj, dict):
        if "results" in obj and isinstance(obj["results"], list) and len(obj["results"]) == 1:
            return _compact(obj["results"][0], depth)
        return {k: _compact(v, depth + 1) for k, v in obj.items()
                if k not in ("version", "error", "message")}
    if isinstance(obj, list):
        return [_compact(x, depth + 1) for x in obj[:12]]
    return obj


def build_evidence(client: httpx.Client, row: dict) -> dict:
    """Budgeted evidence digest for one closed case, from the live graph.

    Uses the deployed agent's own query set keyed off the closed-case row:
    card-testing scorer on the card, new-device/out-of-region scorers on the
    case's transactions, and a 1-hop neighborhood on the first transaction.
    """
    card = (row.get("card_id") or "").strip()
    txn_ids = [t for t in (row.get("txn_ids") or "").split("|") if t.strip()]
    first = (row.get("first_fraud_txn_id") or "").strip() or (txn_ids[0] if txn_ids else "")
    probe_txns = ([first] if first else []) + [t for t in txn_ids if t != first][:2]
    probe_txns = [t for t in probe_txns if t][:3]

    ev: dict = {"card_id": card, "n_case_txns": row.get("n_txns", ""),
                "outcome_window": (row.get("opened_at") or "")[:10]}
    ev["card_testing"] = safe(client, "q_card_testing_score", {"card_id": card})
    for i, t in enumerate(probe_txns):
        ev[f"new_device_cnp[{i}]"] = safe(client, "q_new_device_cnp_score", {"txn_id": t})
        ev[f"out_of_region[{i}]"] = safe(client, "q_out_of_region_score", {"txn_id": t})
    if first:
        ev["neighborhood"] = safe(client, "q_khop_expand",
                                  {"start_txn": first, "max_hops": 1})
    return ev


def build_prompt(case: dict, evidence: dict) -> str:
    digest = json.dumps(_compact(evidence))[:6000]
    return (
        "You are a senior bank fraud analyst working inside an agentic investigation "
        "platform backed by a TigerGraph fraud graph. Investigate the case below.\n\n"
        f"{POLICY}\nCASE: {json.dumps(case)}\n\n"
        f"EVIDENCE DIGEST (compact tool outputs from the graph): {digest}\n\n"
        "Think step by step in <=12 short numbered steps: state hypotheses, which policy "
        "rules could fire, what evidence confirms/refutes each, your calibrated "
        "fraud_probability, whether to gather more evidence (EVSI) or act, and the "
        "stopping reason. Then output the final answer as a single JSON object with "
        "exactly these keys: "
        '{"case_id", "verdict" ("fraud"|"legitimate"|"uncertain"), "fraud_probability" '
        "(0-1, calibrated: confirmed fraud >= 0.85, legitimate <= 0.15, be honest), "
        '"pattern" (card_testing|card_not_present_fraud|card_not_present_new_device|'
        "out_of_region_use|account_takeover|undocumented|none), "
        '"pattern_description" (only when undocumented), "affected_txn_ids" (strings), '
        '"first_suspicious_txn_id", "connected_card_ids" (strings), '
        '"connected_device_profiles" (list of device fingerprint strings, same '
        'format as DeviceInfo|OS|browser|screen), "exposure_usd", '
        '"evidence" (list of {claim, source: graph|document|customer|external, '
        "ref, entity_ids}), \"similar_prior_cases\" (CC-xxxx ids), \"summary\" (2-4 "
        'sentences), "next_best_actions" (list of {action, route: auto|L1|L2, reason '
        "citing rule ids}), \"sar\" ({\"file\": bool, \"reason\" citing rules, "
        '"narrative" only when filing: who/what/when/where/how/exposure, "subjects", '
        '"total_amount_usd", "activity_dates": [first_date, last_date]}), "stop_reason".'
    )


def ask_gemini(client: httpx.Client, auth: TokenRefresher, prompt: str) -> str:
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/"
        f"{LOCATION}/publishers/google/models/{MODEL}:generateContent"
    )
    # 8192 output tokens: fraud answers carry a full SAR narrative (who/what/
    # when/where/how + 6-12 sentences) plus 4-6 recommended actions. At 4096 the
    # JSON was truncated mid-object for ~65% of fraud cases, which is why the
    # first scaled run produced 17 usable fraud traces out of 900.
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 8192}}
    for attempt in range(5):
        r = client.post(url, json=body,
                        headers={"Authorization": f"Bearer {auth.get(force=attempt >= 3)}"},
                        timeout=300)
        if r.status_code in (429, 500, 503) and attempt < 4:
            time.sleep(15 * (attempt + 1))  # quota backoff
            continue
        r.raise_for_status()
        break
    data = r.json()
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def stratified_sample() -> list:
    """ALL cleared cases + fraud cases stratified by pattern (cap MAX_FRAUD).

    The 900 cleared cases are the discrimination signal the v1 run threw away;
    taking every one balances the curriculum toward the base rate the model
    must actually learn to separate.
    """
    rows = list(csv.DictReader(open(CC_CSV)))
    cleared = [r for r in rows if r["outcome"] == "cleared" and r["txn_ids"].strip()]
    fraud = [r for r in rows if r["outcome"] == "confirmed_fraud" and r["txn_ids"].strip()]
    rng = random.Random(SEED)

    by_pattern: dict = defaultdict(list)
    for r in fraud:
        by_pattern[r["pattern"] or "unknown"].append(r)
    per_pattern = max(1, MAX_FRAUD // max(1, len(by_pattern)))
    sample_fraud = []
    for pat, group in sorted(by_pattern.items()):
        rng.shuffle(group)
        sample_fraud.extend(group[:per_pattern])
    rng.shuffle(sample_fraud)
    sample_fraud = sample_fraud[:MAX_FRAUD]

    # Rare patterns (card_testing: 16, undocumented: 9 in this dataset) get
    # trial upweighting — 4 teacher samples per case — so they are learnable
    # despite their scarcity. Common patterns get 1 sample per case.
    RARE = {"card_testing", "undocumented"}
    rare_ids = {r["case_id"] for r in sample_fraud if r["pattern"] in RARE}

    print(f"stratified: cleared={len(cleared)} fraud={len(sample_fraud)} "
          f"(patterns={ {k: min(len(v), per_pattern) for k, v in sorted(by_pattern.items())} })",
          flush=True)
    cases = []
    for r in cleared + sample_fraud:
        trials = [0] if r["case_id"] not in rare_ids else [0, 1, 2, 3]
        for t in trials:
            cases.append({
                "case_id": r["case_id"],
                "customer_id": r["customer_id"],
                "card_id": r["card_id"],
                "opened_at": r["opened_at"],
                "trigger_text": (
                    f"Investigation file {r['case_id']} on card {r['card_id']} "
                    f"(customer {r['customer_id']}), opened {r['opened_at'][:10]}. "
                    "Review the card's activity in this window and decide."
                ),
                "_outcome": r["outcome"],
                "_pattern": r["pattern"],
                "_txn_ids": r["txn_ids"],
                "_first": r["first_fraud_txn_id"],
                "_exposure": r["exposure_usd"],
                "_trial": t,
            })
    rng.shuffle(cases)
    return cases


WRITE_LOCK = threading.Lock()
done_keys: set = set()


def process(case: dict, client: httpx.Client, auth: TokenRefresher) -> int:
    key = (case["case_id"], case["_trial"])
    if key in done_keys:
        return 0
    try:
        evidence = build_evidence(client, case)
        prompt = build_prompt(case, evidence)
        reply = ask_gemini(client, auth, prompt)
        rec = {
            "case_id": case["case_id"],
            "trial": case["_trial"],
            "outcome": case["_outcome"],
            "pattern": case["_pattern"],
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": reply}],
            "tool_trace": ["q_card_testing_score", "q_new_device_cnp_score",
                           "q_out_of_region_score", "q_khop_expand"],
        }
        with WRITE_LOCK:
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done_keys.add(key)
        return 1
    except Exception as e:
        with WRITE_LOCK:
            print(f"FAIL {key}: {str(e)[:140]}", flush=True)
        return 0


def main() -> int:
    cases = stratified_sample()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                rec = json.loads(line)
                done_keys.add((rec["case_id"], rec.get("trial", 0)))
            except Exception:
                pass
    todo = [c for c in cases if (c["case_id"], c["_trial"]) not in done_keys]
    print(f"total={len(cases)} done={len(done_keys)} todo={len(todo)}", flush=True)
    if not todo:
        print("SCALED_GEN_DONE", flush=True)
        return 0

    client = httpx.Client()
    auth = TokenRefresher(client)
    n_ok = n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process, c, client, auth): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            if fut.result():
                n_ok += 1
            else:
                n_fail += 1
            if i % 25 == 0 or i == len(todo):
                rate = i / max(1e-9, time.time() - t0)
                eta_min = (len(todo) - i) / max(1e-9, rate) / 60
                print(f"progress {i}/{len(todo)} ok={n_ok} fail={n_fail} "
                      f"rate={rate:.2f}/s eta={eta_min:.0f}min", flush=True)
    print(f"SCALED_GEN_DONE ok={n_ok} fail={n_fail}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())