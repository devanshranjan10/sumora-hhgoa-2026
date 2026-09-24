"""Bounded graph tools for the investigation loop.

The agent exposes a small set of domain queries. Selected reads use the
official TigerGraph MCP installed-query tool; writes and other reads use
RESTPP. Results are bounded before they reach the answer or trace.

Modes:
  live - calls TigerGraph installed queries through RESTPP and official MCP
  stub - deterministic features from the local CSV index (bench parity)
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from bench.dataset import DatasetIndex, norm_txn_id, txn_id_variants

# The primitive names (the stable tool surface the LLM/agent sees).
PRIMITIVES = [
    "q_evidence_pack",           # budgeted case evidence (vertices+edges)
    "q_khop_expand",             # bounded k-hop neighborhood
    "q_card_testing_score",
    "q_new_device_cnp_score",
    "q_out_of_region_score",
    "q_ato_score",
    "q_mule_fanin_score",
    "q_case_structural_similarity",
    "q_recurring_entities",
    "q_hypothesis_balance",
    "q_case_feature_vector",
    "q_community_fingerprint",
    "q_anomalous_communities",
    "action:verify_with_customer",  # via actions_api (persona in bench mode)
]

MCP_READ_QUERIES = frozenset({
    "q_card_testing_score", "q_new_device_cnp_score", "q_out_of_region_score",
    "q_ato_score", "q_case_structural_similarity",
})


@dataclass
class ToolResult:
    name: str
    tokens: int
    data: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str = ""


class InvestigationTools:
    """Facade over MCP and RESTPP in live mode, or the CSV index offline."""

    def __init__(self, idx: DatasetIndex, metrics: Any = None,
                 mode: str = "stub", host: Optional[str] = None,
                 memory_mode: str = "full",
                 neutralize_graph: bool = False) -> None:
        self.idx = idx
        self.metrics = metrics
        self.mode = mode
        # memory_mode: "full" | "withheld" | "shuffled:<seed>" (counterfactuals)
        self.memory_mode = memory_mode
        # neutralize_graph: ablation - collapse structural signal to "nothing seen"
        self.neutralize_graph = neutralize_graph
        self.host = host or os.environ.get("TG_HOST", "127.0.0.1")
        self.port = int(os.environ.get("TG_PORT", "9000"))
        self._client: Optional[httpx.Client] = None
        self.mcp_calls: List[Dict[str, Any]] = []

    # -- plumbing ---------------------------------------------------------
    def _record(self, name: str, tokens: int) -> None:
        if self.metrics is not None:
            self.metrics.tool_call(name, tokens)

    def _gsql(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        # TigerGraph RESTPP URL-decodes request values: a literal '%' arrives
        # server-side as the start of a %-escape ("90% CI" stored as "90\fI").
        # Pre-escape '%' -> '%25' so stored values match what we send.
        safe = {k: (v.replace("%", "%25") if isinstance(v, str) else v)
                for k, v in params.items()}
        # Answers use T-prefixed IDs; Transaction vertices use raw CSV IDs.
        if "txn_id" in safe:
            safe["txn_id"] = txn_id_variants(str(safe["txn_id"]))[1]
        if "affected_txn_ids" in safe:
            safe["affected_txn_ids"] = [txn_id_variants(str(t))[1]
                                        for t in safe["affected_txn_ids"]]
        if name in MCP_READ_QUERIES:
            from agent.mcp_graph import run_installed_query
            response = run_installed_query(name, safe)
            rows = response.get("results", [])
            preview = json.dumps(rows[:1], ensure_ascii=False)
            self.mcp_calls.append({
                "tool": "tigergraph__run_installed_query",
                "query_name": name,
                "parameters": safe,
                "row_count": len(rows),
                "result_preview": preview[:4000],
                "truncated": len(preview) > 4000 or len(rows) > 1,
            })
            return response
        url = f"http://{self.host}:{self.port}/query/SumoraFraudGraph/{name}"
        r = httpx.post(url, json=safe, timeout=90)
        r.raise_for_status()
        return r.json()

    def _rows(self, name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Unwrap TG's RESTPP envelope to the query's PRINTed rows.

        GSQL 4.x PRINT output arrives as {"results": [row, ...]} where each row
        maps PRINT aliases to values. All live adapters normalize through this.
        """
        env = self._gsql(name, params)
        res = env.get("results", [])
        return res if isinstance(res, list) else [res]

    def _stub_card_txns(self, card_id: str, window: int = 20) -> List[str]:
        return list(self.idx.card_txns.get(card_id, []))[:window]

    # -- primitives -------------------------------------------------------
    def q_evidence_pack(self, case_id: str, flagged_txn_id: str,
                        card_id: str, max_vertices: int = 30,
                        max_edges: int = 60) -> ToolResult:
        self._record("q_evidence_pack", 260)
        if self.mode == "live":
            try:
                rows = self._rows("q_evidence_pack", {"case_id": case_id,
                                                      "max_vertices": max_vertices,
                                                      "max_edges": max_edges})
                r0 = rows[0] if rows else {}
                # Graph rows are "txn_id|amt|risk|fraud_label|channel|ts".
                txns: List[Dict[str, Any]] = []
                for raw in r0.get("transactions", []):
                    parts = str(raw).split("|")
                    if len(parts) >= 6:
                        txns.append({"txn_id": norm_txn_id(parts[0]), "amount_usd": float(parts[1]),
                                     "risk": float(parts[2]), "fraud_label": parts[3],
                                     "channel": parts[4], "ts": parts[5]})
                t = norm_txn_id(flagged_txn_id)
                anchor = next((x for x in txns if x["txn_id"] == t), None)
                if anchor is None:
                    # Anchor always present (pack starts from INVESTIGATES); fall
                    # back to the local index rather than emitting a blank claim.
                    anchor = {"txn_id": t, "amount_usd": self.idx.amount(t),
                              "channel": "in_person" if not self.idx.txn_online.get(t, True) else "online",
                              "ts": "", "risk": 0.0, "fraud_label": ""}
                ch = anchor.get("channel", "")
                pack = {
                    "flagged_txn": {"txn_id": t,
                                    "amount_usd": anchor["amount_usd"],
                                    "date": anchor.get("ts", ""),
                                    "online": "in_person" not in ch,
                                    "device": self.idx.txn_device.get(t, "")},
                    "card_window": [x["txn_id"] for x in txns[:10]],
                    "ranked_txns": txns[:max_vertices],
                    "device_shared_cases": sum(
                        1 for cc in self.idx.closed_case_ids
                        if any(self.idx.txn_device.get(x) == self.idx.txn_device.get(t, "")
                               for x in self.idx.closed_case_txns.get(cc, ()))
                    ) if self.idx.txn_device.get(t, "") else 0,
                    "entities": {k: r0.get(k, []) for k in
                                 ("card_ids", "device_ids", "email_ids",
                                  "cluster_ids", "region_ids")},
                    "edges": r0.get("edges", [])[:max_edges],
                    "limits": {"max_vertices": max_vertices, "max_edges": max_edges},
                }
                return ToolResult("q_evidence_pack", 260, pack)
            except Exception as e:  # fall back to stub features
                return ToolResult("q_evidence_pack", 260, {}, ok=False, error=str(e)[:160])
        t = norm_txn_id(flagged_txn_id)
        device = self.idx.txn_device.get(t, "")
        pack = {
            "flagged_txn": {"txn_id": flagged_txn_id,
                            "amount_usd": self.idx.amount(t),
                            "date": self.idx.date_of(t),
                            "online": self.idx.txn_online.get(t, True),
                            "device": device},
            "card_window": self._stub_card_txns(card_id)[:10],
            "device_shared_cases": sum(
                1 for cc in self.idx.closed_case_ids
                if any(self.idx.txn_device.get(x) == device
                       for x in self.idx.closed_case_txns.get(cc, ()))
            ) if device else 0,
            "limits": {"max_vertices": max_vertices, "max_edges": max_edges},
        }
        return ToolResult("q_evidence_pack", 260, pack)

    def q_card_testing_score(self, card_id: str, flagged_txn_id: str = "") -> ToolResult:
        self._record("q_card_testing_score", 180)
        cutoff = int(self.idx.txn_dt[norm_txn_id(flagged_txn_id)]) if flagged_txn_id else 0
        if self.mode == "live":
            try:
                row = (self._rows("q_card_testing_score", {
                    "card_id": card_id, "cutoff_dt": cutoff,
                }) or [{}])[0]
                return ToolResult("q_card_testing_score", 180, {
                    "card_id": card_id,
                    "small_auths": int(row.get("probes", 0)),
                    "probe_to_large": int(row.get("probe_to_large_transitions", 0)),
                    "score": float(row.get("card_testing_score", 0.0)),
                })
            except Exception as exc:
                raise RuntimeError("TigerGraph MCP card-testing query failed") from exc
        # Stub: count small-auth-then-large sequences in the card window.
        txns = list(self.idx.card_txns.get(card_id, []))
        if cutoff:
            txns = [t for t in txns if cutoff - 3600 <= self.idx.txn_dt[t] <= cutoff]
        else:
            txns = txns[-40:]
        txns = sorted(txns, key=lambda t: self.idx.txn_dt[t])
        small = [t for t in txns if 0 < self.idx.amount(t) <= 5.0
                 and self.idx.txn_online.get(t, False)]
        transitions = sum(
            self.idx.amount(a) <= 5.0 and self.idx.txn_online.get(a, False)
            and self.idx.amount(b) > 50.0
            and self.idx.txn_dt[b] - self.idx.txn_dt[a] <= 3600
            for a, b in zip(txns, txns[1:])
        )
        score = min(1.0, len(small) / 4.0 + transitions * .25) if small else 0.0
        return ToolResult("q_card_testing_score", 180,
                          {"card_id": card_id, "small_auths": len(small),
                           "probe_to_large": transitions, "score": score})

    def q_new_device_cnp_score(self, flagged_txn_id: str) -> ToolResult:
        self._record("q_new_device_cnp_score", 170)
        t = norm_txn_id(flagged_txn_id)
        if self.mode == "live":
            try:
                row = (self._rows("q_new_device_cnp_score", {"txn_id": t}) or [{}])[0]
                device_observed = float(row.get("w_device_novel", 0.0)) > 0
                new_dev = device_observed and int(row.get("prior_txns_same_device", 0)) == 0
                return ToolResult("q_new_device_cnp_score", 170, {
                    "txn_id": flagged_txn_id,
                    "new_device": new_dev,
                    "device_observed": device_observed,
                    "cnp": self.idx.txn_online.get(t, True),
                    "score": float(row.get("new_device_cnp_score", 0.0)),
                    "support_count": int(row.get("support_count", 0)),
                    "contradict_count": int(row.get("contradict_count", 0)),
                    "w_device_shared": float(row.get("w_device_shared", 0.0)),
                    "w_email_mismatch": float(row.get("w_email_mismatch", 0.0)),
                    "w_amount_outlier": float(row.get("w_amount_outlier", 0.0)),
                })
            except Exception as e:
                return ToolResult("q_new_device_cnp_score", 170, {}, ok=False, error=str(e)[:160])
        device = self.idx.txn_device.get(t, "")
        online = self.idx.txn_online.get(t, True)
        prior_use = sum(1 for tx, d in self.idx.txn_device.items() if d == device and tx != t)
        score = (0.6 if online else 0.2) + (0.4 if prior_use == 0 else 0.0)
        return ToolResult("q_new_device_cnp_score", 170,
                          {"txn_id": flagged_txn_id, "new_device": prior_use == 0,
                           "cnp": online, "score": min(1.0, score)})

    def q_out_of_region_score(self, flagged_txn_id: str) -> ToolResult:
        self._record("q_out_of_region_score", 170)
        t = norm_txn_id(flagged_txn_id)
        if self.mode == "live":
            try:
                row = (self._rows("q_out_of_region_score", {"txn_id": t}) or [{}])[0]
                return ToolResult("q_out_of_region_score", 170, {
                    "txn_id": flagged_txn_id,
                    "score": float(row.get("out_of_region_score", 0.0)),
                })
            except Exception as e:
                return ToolResult("q_out_of_region_score", 170, {}, ok=False, error=str(e)[:160])
        device = self.idx.txn_device.get(t, "")
        shared_regions = sum(1 for tx, d in self.idx.txn_device.items()
                             if d == device and tx != t)
        score = 0.7 if shared_regions == 0 else 0.2
        return ToolResult("q_out_of_region_score", 170,
                          {"txn_id": flagged_txn_id, "score": score})

    def q_ato_score(self, customer_id: str) -> ToolResult:
        self._record("q_ato_score", 170)
        if self.mode == "live":
            try:
                row = (self._rows("q_ato_score", {"customer_id": customer_id}) or [{}])[0]
                return ToolResult("q_ato_score", 170, {
                    "customer_id": customer_id,
                    "n_cards": int(row.get("n_cards", 0)),
                    "score": float(row.get("ato_score", 0.0)),
                })
            except Exception as e:
                return ToolResult("q_ato_score", 170, {}, ok=False, error=str(e)[:160])
        cards = self.idx.customer_cards.get(customer_id, [])
        return ToolResult("q_ato_score", 170,
                          {"customer_id": customer_id, "n_cards": len(cards), "score": 0.0})

    def q_mule_fanin_score(self, entity_id: str) -> ToolResult:
        self._record("q_mule_fanin_score", 180)
        if self.mode == "live":
            try:
                row = (self._rows("q_mule_fanin_score", {"entity_id": entity_id}) or [{}])[0]
                fan_in = int(row.get("fan_in", row.get("in_count", 0)))
                return ToolResult("q_mule_fanin_score", 180,
                                  {"entity_id": entity_id, "fan_in": fan_in,
                                   "score": min(1.0, fan_in / 8.0)})
            except Exception as e:
                return ToolResult("q_mule_fanin_score", 180, {}, ok=False, error=str(e)[:160])
        n = len(self.idx.card_customer.get(entity_id, []))
        return ToolResult("q_mule_fanin_score", 180,
                          {"entity_id": entity_id, "fan_in": n, "score": min(1.0, n / 8.0)})

    def q_case_structural_similarity(self, device: str, limit: int = 3,
                                     case_id: str = "") -> ToolResult:
        self._record("q_case_structural_similarity", 220)
        if self.mode == "live" and case_id:
            try:
                row = (self._rows("q_case_structural_similarity",
                                  {"case_id": case_id, "top_k": max(limit, 10)}) or [{}])[0]
                def ranked(values: List[str]) -> List[str]:
                    scored = []
                    for value in values:
                        parts = str(value).split("|")
                        if len(parts) >= 2:
                            scored.append((parts[0], float(parts[1])))
                    return [case for case, _ in sorted(
                        scored, key=lambda item: (-item[1], item[0]))[:limit]]

                closed_top = ranked(row.get("similar_closed_cases", []))
                n_fraud = sum(1 for cc in closed_top
                              if cc in self.idx.closed_case_outcome_fraud)
                return ToolResult("q_case_structural_similarity", 220, {
                    "similar_prior_cases": closed_top,
                    "agent_cases": ranked(row.get("similar_cases", [])),
                    "similar_closed_cases": closed_top,
                    "n_fraud": n_fraud,
                    "n_cleared": max(0, len(closed_top) - n_fraud),
                })
            except Exception as e:
                return ToolResult("q_case_structural_similarity", 220,
                                  {}, ok=False, error=str(e)[:160])
        similar: List[str] = []
        if device:
            for cc in self.idx.closed_case_ids:
                if any(self.idx.txn_device.get(t) == device
                       for t in self.idx.closed_case_txns.get(cc, ())):
                    similar.append(cc)
        similar = sorted(similar)[:limit]
        return ToolResult("q_case_structural_similarity", 220,
                          {"similar_prior_cases": similar,
                           "n_fraud": len(similar), "n_cleared": 0})

    def q_case_feature_vector(self, flagged_txn_id: str, card_id: str,
                              customer_id: str, prior_score: float | None = None) -> ToolResult:
        """The deterministic feature vector that feeds the calibrator.

        Memory features (mem_fraud/mem_cleared) come from the same device-keyed
        maps the calibrator was fit with (agent.memory) - signed, bank-confirmed
        cases only.
        """
        self._record("q_case_feature_vector", 240)
        from agent.features import feature_vector
        data = feature_vector(self.idx, flagged_txn_id, prior_score,
                              self.memory_mode, self.neutralize_graph)
        return ToolResult("q_case_feature_vector", 240, data)

    def q_hypothesis_balance(self, p: float) -> ToolResult:
        """Reports which hypothesis the current evidence favors (stub: from p)."""
        self._record("q_hypothesis_balance", 150)
        favored = "fraud" if p >= 0.5 else "legitimate"
        return ToolResult("q_hypothesis_balance", 150,
                          {"favored": favored, "p": p,
                           "margin": abs(p - 0.5) * 2})

    def q_recurring_entities(self, card_id: str, flagged_txn_id: str) -> ToolResult:
        self._record("q_recurring_entities", 190)
        flagged = norm_txn_id(flagged_txn_id)
        cutoff = self.idx.txn_dt[flagged]
        observed = Counter(
            self.idx.txn_device[txn]
            for txn in self.idx.card_txns.get(card_id, [])
            if txn != flagged and self.idx.txn_dt[txn] < cutoff
            and self.idx.txn_device.get(txn)
        )
        recurring = [device for device, count in observed.most_common()
                     if count >= 4][:3]
        return ToolResult("q_recurring_entities", 190, {"recurring": recurring})

    def q_community_fingerprint(self, community_id: int) -> ToolResult:
        self._record("q_community_fingerprint", 210)
        return ToolResult("q_community_fingerprint", 210,
                          {"community_id": community_id, "available": self.mode == "live"})

    def q_anomalous_communities(self) -> ToolResult:
        """Return the shipped undocumented typology when discovery passed."""
        self._record("q_anomalous_communities", 260)
        from agent.discovery import load_discovery, shipped_typology
        typ = shipped_typology()
        disc = load_discovery()
        if typ is None:
            return ToolResult("q_anomalous_communities", 260, {
                "available": True, "significant": False,
                "rings": 0, "perm_p_value": disc.get("perm_p_value"),
            })
        demo = disc.get("demo_exemplar") or {}
        return ToolResult("q_anomalous_communities", 260, {
            "available": True,
            "significant": True,
            "perm_p_value": disc.get("perm_p_value"),
            "observed_ring_count": disc.get("observed_ring_count"),
            "typology": typ,
            "demo_exemplar": demo,
        })

    # -- customer interaction (via actions_api persona in bench mode) -----
    def record_case(self, case_id: str, status: str, verdict: str, p: float,
                    pattern: str, pattern_desc: str,
                    affected_txn_ids: List[str], exposure_usd: float,
                    summary: str, stop_reason: str,
                    similar_case_ids: List[str], evidence: List[Dict[str, Any]],
                    initial_actions: List[Dict[str, Any]],
                    final_actions: List[Dict[str, Any]], what_changed: str,
                    steps: List[Dict[str, Any]]) -> bool:
        """Write the resolved case back into the graph (S6 mandated capability).

        live mode: q_record_case upserts the FraudCase decision fields and
        links affected txns / matched pattern / cited priors; per-item
        q_add_evidence / q_add_decision / q_add_step persist the audit trail.
        stub mode: trace-only, returns False - the answer then honestly says
        written_to_graph=false (no graph to write to).
        """
        self._record("q_record_case", 200)
        if self.mode != "live":
            return False
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            self._gsql("q_record_case", {
                "case_id": case_id, "status": status, "verdict": verdict,
                "p_val": p, "pattern": pattern, "pattern_desc": pattern_desc,
                "exposure_usd": exposure_usd, "summary": summary,
                "stop_reason": stop_reason, "closed_ts": ts,
                "affected_txn_ids": list(affected_txn_ids),
                "similar_case_ids": list(similar_case_ids),
            })
            for i, ev in enumerate(evidence, 1):
                self._gsql("q_add_evidence", {
                    "case_id": case_id, "evidence_id": f"EV-{case_id}-{i:02d}",
                    "claim": str(ev.get("claim", "")),
                    "source": str(ev.get("source", "graph")),
                    "ref": str(ev.get("ref", "")),
                    "entity_ids_json": json.dumps(ev.get("entity_ids", [])),
                })
            for phase, acts in (("initial", initial_actions), ("final", final_actions)):
                self._gsql("q_add_decision", {
                    "case_id": case_id, "decision_id": f"DEC-{case_id}-{phase}",
                    "phase": phase, "actions_json": json.dumps(acts),
                    "what_changed": what_changed if phase == "final" else "",
                    "ts": ts,
                })
            for i, st in enumerate(steps, 1):
                # step dicts are {step: <node-name>, ...payload}; the index is
                # the enumerate position, not st["step"] (a node-name string).
                self._gsql("q_add_step", {
                    "case_id": case_id, "step_id": f"{case_id}:{i:04d}",
                    "step_index": i,
                    "node": str(st.get("step", "")),
                    "tool_call_json": json.dumps(st.get("tool", "")),
                    "result_ref": str(st.get("result", ""))[:200],
                    "reasoning": str(st.get("reasoning", "")),
                    "rules_json": json.dumps(st.get("rules", [])),
                    "ts": ts,
                })
            return True
        except Exception as e:
            # A failed mandated write must never be silent: print the reason
            # (the answer still honestly reports written_to_graph=false).
            import traceback
            print(f"[record_case] write-back FAILED for {case_id}: {e!r}",
                  file=sys.stderr)
            traceback.print_exc()
            self._record("q_record_case:FAILED", 0)
            return False

    def verify_with_customer(self, case_id: str, question: str,
                             persona: Any = None, context: Any = None) -> ToolResult:
        self._record("action:verify_with_customer", 120)
        if persona is None:
            return ToolResult("action:verify_with_customer", 120, {"response": "no_reply"})
        reply = persona.respond(case_id, "customer_validation", question, context=context)
        return ToolResult("action:verify_with_customer", 120, reply)
