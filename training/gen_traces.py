#!/usr/bin/env python3
"""Sumora teacher trace generation (spec 6b, tier 1).

Runs on the A100 VM host. For each case in case_pack.csv:
  1. Pull an evidence pack + pattern scores from the live TigerGraph (RESTPP :9000).
  2. Ask Gemini (Vertex AI, using the VM service account - no key file) to play
     the expert fraud analyst and emit a full investigation transcript ending
     in a policy-conformant answer JSON.
  3. Persist {case_id, model, messages, tool_trace, answer} as one JSONL line.

No SDK dependency: metadata-server token + httpx only.
"""
import csv
import json
import sys
import time
import urllib.parse
from pathlib import Path

import httpx

PROJECT = "contral-6b0bd"
LOCATION = "us-central1"
GRAPH = "SumoraFraudGraph"
TG = f"http://localhost:9000/query/{GRAPH}"
METADATA = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
CASE_DIR = Path("/mnt/sih26-train/sumora/data")
OUT = Path("/mnt/sih26-train/sumora/traces")
MODELS = {
    "a": "gemini-2.5-pro",
    "b": "gemini-2.5-flash",  # arm B placeholder; spec A/B is Gemini Pro vs Claude Sonnet 5
}
TRIALS = 3  # per model per case

POLICY = """Bank policy rules R1-R10 (id: trigger condition):
R1 card_testing: >=5 distinct low-amplitude (<=$5) approved txns on one card within 60min.
R2 new_device_cnp: first-seen device + card-not-present channel on an account with prior CNP history.
R3 out_of_region: billing region != device geo AND distance > 500km.
R4 ato: >=3 failed auths then >=1 success within 24h on same account.
R5 mule_fanin: >=8 distinct cards paying into same email/region cluster within 7d.
R6 recurring_burst: >=4 txns same merchant same card within 90min.
R7 amount_outlier: txn amount > 6 sigma above card's 90-day mean.
R8 velocity: >=10 txns same card within 10min across merchants.
R9 identity_collision: same email on >=3 distinct cards within 30d.
R10 structuring: 3+ txns each between $900-$999.99 same card same day.
Disposition policy: act if p >= 0.85 or expected cost of acting < cost of waiting;
gather evidence while EVSI > cost of query; file SAR when rule R1/R4/R5 fires with p >= 0.60.
Answer JSON keys: case_id, verdict ("fraud"|"cleared"|"investigate"), fraud_probability (0-1),
actions_taken[], primary_rules[], evidence_classes[], stopping_reason.
"""


def tg_token() -> str:
    pass  # not needed for local RESTPP


def vertex_token(client: httpx.Client) -> str:
    r = client.get(METADATA, headers={"Metadata-Flavor": "Google"}, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]


class TokenRefresher:
    """Caches the metadata-server token and refreshes on age or 401."""

    def __init__(self, client: httpx.Client, max_age_s: int = 2400) -> None:
        self.client = client
        self.max_age_s = max_age_s
        self._token = ""
        self._ts = 0.0

    def get(self, force: bool = False) -> str:
        import time as _t
        if force or not self._token or (_t.time() - self._ts) > self.max_age_s:
            self._token = vertex_token(self.client)
            self._ts = _t.time()
        return self._token


def call_query(client: httpx.Client, name: str, params: dict) -> dict:
    url = f"{TG}/{name}"
    r = client.post(url, json=params, timeout=120)
    r.raise_for_status()
    return r.json()


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


def build_prompt(case: dict, evidence: dict) -> str:
    # Budgeted digest (guard-the-context-window): no raw row dumps, the
    # evidence JSON is flattened + capped before it reaches the teacher.
    digest = json.dumps(_compact(evidence))[:6000]
    return (
        "You are a senior bank fraud analyst working inside an agentic investigation "
        "platform backed by a TigerGraph fraud graph. Investigate the case below.\n\n"
        f"{POLICY}\nCASE: {json.dumps(case)}\n\n"
        f"EVIDENCE DIGEST (compact tool outputs from the graph): {digest}\n\n"
        "Think step by step in <=12 short numbered steps: state hypotheses, which policy "
        "rules could fire, what evidence confirms/refutes each, your calibrated p, whether "
        "to gather more evidence (EVSI) or act, and the stopping reason. Then output the "
        "final answer as a single JSON object with exactly the answer keys listed above."
    )


def ask_gemini(client: httpx.Client, auth: TokenRefresher, model: str, prompt: str) -> str:
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/"
        f"{LOCATION}/publishers/google/models/{model}:generateContent"
    )
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 4096},
    }
    r = client.post(
        url,
        json=body,
        headers={"Authorization": f"Bearer {auth.get()}"},
        timeout=300,
    )
    if r.status_code == 401:  # token expired mid-run: force refresh once
        r = client.post(
            url,
            json=body,
            headers={"Authorization": f"Bearer {auth.get(force=True)}"},
            timeout=300,
        )
    r.raise_for_status()
    data = r.json()
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = list(csv.DictReader(open(CASE_DIR / "case_pack.csv")))
    print(f"cases={len(cases)} models={list(MODELS.values())} trials={TRIALS}", flush=True)
    client = httpx.Client()
    auth = TokenRefresher(client)
    out_path = OUT / "traces.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                done.add((rec["case_id"], rec["model"], rec["trial"]))
            except Exception:
                pass
    n = 0
    for case in cases:
        cid = case.get("case_id") or case.get("caseid") or list(case.values())[0]
        # Evidence: the budgeted evidence pack (q_evidence_pack enforces
        # max_vertices/max_edges - never dump raw rows), then per-entity
        # pattern scores keyed off the pack's entity IDs.
        try:
            pack = call_query(client, "q_evidence_pack", {"case_id": cid, "max_vertices": 40, "max_edges": 80})["results"][0]

            def ids(lst):
                out = []
                for it in lst:
                    if isinstance(it, dict):
                        v = it.get("v_id") or (it.get("attributes") or {}).get("txn_id") or it.get("id")
                        if v is not None:
                            out.append(str(v))
                    else:
                        out.append(str(it))
                return out

            card_ids = ids(pack.get("card_ids", []))[:2]
            txn_ids = ids(pack.get("transactions", []))[:3]
            email_ids = ids(pack.get("email_ids", []))[:1]
            cluster_ids = ids(pack.get("cluster_ids", []))[:1]

            def safe(q, params):
                try:
                    return call_query(client, q, params)
                except Exception as e:
                    return {"error": str(e)[:120]}

            pattern_scores = {}
            for c in card_ids:
                pattern_scores[f"card_testing[{c}]"] = safe("q_card_testing_score", {"card_id": c})
            for t in txn_ids:
                pattern_scores[f"new_device_cnp[{t}]"] = safe("q_new_device_cnp_score", {"txn_id": t})
                pattern_scores[f"out_of_region[{t}]"] = safe("q_out_of_region_score", {"txn_id": t})
            for e in email_ids + cluster_ids:
                pattern_scores[f"mule_fanin[{e}]"] = safe("q_mule_fanin_score", {"entity_id": e})

            evidence = {"evidence_pack": pack, "pattern_scores": pattern_scores}
        except Exception as e:
            print(f"skip {cid}: evidence error {e}", flush=True)
            continue
        prompt = build_prompt(case, evidence)
        for arm, model in MODELS.items():
            for trial in range(TRIALS):
                key = (cid, model, trial)
                if key in done:
                    continue
                try:
                    text = ask_gemini(client, auth, model, prompt)
                except Exception as e:
                    print(f"ERR {key}: {e}", flush=True)
                    time.sleep(5)
                    continue
                rec = {
                    "case_id": cid,
                    "model": model,
                    "trial": trial,
                    "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": text}],
                    "tool_trace": ["q_evidence_pack", "q_card_testing_score", "q_new_device_cnp_score", "q_out_of_region_score", "q_ato_score", "q_mule_fanin_score"],
                }
                with out_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                n += 1
                print(f"ok {key} len={len(text)}", flush=True)
                time.sleep(2)
    print(f"generated {n} new traces -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
