"""AgentRunner protocol + StubAgentRunner (spec section 9).

``AgentRunner`` is the seam between the benchmark harness and the real
LangGraph agent. The stub produces a realistic, fully policy-consistent answer
from deterministic, data-derived features so the whole pipeline — schema
validation, ID-existence checks, rule-fired checks, answer-file writing,
determinism — is testable end-to-end today, before the agent service exists.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from actions_api import policy
from actions_api.personas import ScriptedPersonaResponder
from bench.dataset import DatasetIndex, norm_txn_id
from bench.instrument import CaseMetrics


@dataclass
class CaseContext:
    case_id: str
    opened_at: str
    trigger_type: str
    trigger_text: str
    flagged_txn_id: str
    card_id: str
    customer_id: str
    risk_score: Optional[float]


@dataclass
class AgentResult:
    answer: Dict[str, Any]
    case_state: Dict[str, Any]  # feature state for rule-fired validation


class AgentRunner(Protocol):
    def investigate(
        self, ctx: CaseContext, idx: DatasetIndex, metrics: CaseMetrics
    ) -> AgentResult:
        ...


class StubAgentRunner:
    """Deterministic, heuristic stand-in for the real agent loop.

    In deterministic mode the seeded persona reply drives the verdict, exactly
    like the real system will: initial NBA -> evidence request -> final NBA.
    """

    def __init__(self, persona: Optional[ScriptedPersonaResponder] = None, seed: int = 42):
        self.seed = seed
        self.persona = persona or ScriptedPersonaResponder(seed=seed)

    # -- graph/tool call simulation --------------------------------------
    def _q(self, metrics: CaseMetrics, name: str, tokens: int) -> None:
        metrics.tool_call(name, tokens)

    def investigate(
        self, ctx: CaseContext, idx: DatasetIndex, metrics: CaseMetrics
    ) -> AgentResult:
        rng = random.Random(f"{self.seed}:{ctx.case_id}")
        flagged = norm_txn_id(ctx.flagged_txn_id)
        amount = idx.amount(flagged)
        date = idx.date_of(flagged)
        device = idx.txn_device.get(flagged, "")

        # Tool 1-3: fetch flagged txn, card history, device neighborhood.
        self._q(metrics, "query:txn_detail", 180)
        self._q(metrics, "query:card_window", 340)
        self._q(metrics, "query:device_neighbors", 290)

        # Feature extraction from the dataset (deterministic).
        card_history = idx.card_txns.get(ctx.card_id, [])
        history_amts = sorted(abs(idx.amount(t)) for t in card_history if t != flagged)
        typical = history_amts[len(history_amts) // 2] if history_amts else amount
        amount_outlier = bool(history_amts) and amount > max(3.0 * typical, 50.0)
        online = idx.txn_online.get(flagged, True)
        similar: List[str] = []
        if device:
            for cc in idx.closed_case_ids:
                if device and any(idx.txn_device.get(t) == device for t in idx.closed_case_txns.get(cc, ())):
                    similar.append(cc)
        similar = sorted(similar)[:2]
        if similar:
            self._q(metrics, "query:case_structural_similarity", 210)
        connected_cards = sorted(
            {
                c
                for cc in similar
                for c in idx.closed_case_cards.get(cc, ())
                if c != ctx.card_id and idx.card_exists(c)
            }
        )[:3]

        p0 = ctx.risk_score if ctx.risk_score is not None else (0.55 if ctx.trigger_type == "customer_report" else 0.40)
        p = p0
        single_signal = ctx.trigger_type in ("risk_score", "analyst_request")

        # Initial NBA under R1/R8 (before evidence comes back).
        initial: List[Dict[str, str]] = []
        if p < 0.70 and single_signal:
            initial.append({
                "action": "VERIFY_WITH_CUSTOMER", "route": "auto",
                "reason": f"R1: single signal (risk score {p0:.2f}) below 0.70; verify before any block",
            })
        elif p >= 0.85:
            initial.append({
                "action": "DECLINE_TRANSACTION", "route": "L1",
                "reason": "Risk score >= 0.85 on the flagged authorization",
            })
        else:
            initial.append({
                "action": "MONITOR_CARD", "route": "auto",
                "reason": f"Ambiguous signal (p={p0:.2f}); monitor while gathering evidence",
            })

        # Evidence request: the simulated customer reply (README: not provided;
        # we simulate and record the assumption).
        evidence_requests: List[Dict[str, Any]] = []
        reply: Optional[Dict[str, Any]] = None
        if ctx.trigger_type != "customer_report" and p0 < 0.85:
            reply = self.persona.respond(
                ctx.case_id, "customer_validation",
                f"Did you make transaction {flagged} (${amount:.2f})?",
                context={"txn_id": flagged, "amount": amount},
            )
            self._q(metrics, "action:verify_with_customer", 120)
            evidence_requests.append({
                "type": "customer_validation",
                "asked_after_step": 3,
                "assumed_response": reply["response"],
            })

        # Verdict from (stub) evidence fusion.
        customer_denied = ctx.trigger_type == "customer_report" or bool(
            reply and reply.get("denies_transaction")
        )
        customer_confirmed = bool(reply and reply.get("confirmed_transaction"))
        shared_origin = bool(device and connected_cards)

        if customer_confirmed and not amount_outlier:
            verdict, p = "legitimate", round(0.05 + 0.1 * rng.random(), 2)
        elif customer_denied:
            verdict = "fraud"
            p = round(min(0.97, max(p0, 0.80) + 0.06 * rng.random()), 2)
        elif p >= 0.85:
            verdict, p = "fraud", round(min(0.97, p), 2)
        else:
            verdict, p = "uncertain", round(max(0.16, min(0.84, p)), 2)

        exposure = round(abs(amount), 2) if verdict != "legitimate" else 0.0
        affected = [flagged] if verdict != "legitimate" else []

        # Pattern classification (stub heuristic over data features).
        if verdict == "legitimate":
            pattern, pattern_desc = "none", ""
        elif amount_outlier and online and device:
            pattern, pattern_desc = "card_not_present_new_device", ""
        elif online:
            pattern, pattern_desc = "card_not_present_fraud", ""
        else:
            pattern, pattern_desc = "out_of_region_use", ""

        # Final NBA under the policy, given the assumed reply.
        # Rule citations must match actions_api.policy.fired_rules(case_state):
        # only cite a rule whose guard genuinely holds for this case.
        final: List[Dict[str, str]] = []
        if verdict == "legitimate":
            final.append({"action": "CLOSE_NO_FRAUD", "route": "auto",
                          "reason": "R3: customer confirmed the transaction"})
        elif verdict == "fraud":
            block_route = "L2" if exposure > policy.BLOCK_CARD_L2_THRESHOLD_USD else "L1"
            if ctx.trigger_type != "customer_report" and p0 >= 0.85 and not reply:
                final.append({"action": "DECLINE_TRANSACTION", "route": "L1",
                              "reason": "Risk score >= 0.85; decline the pending authorization"})
            if customer_denied:
                block_reason = (f"R2: customer denial; exposure ${exposure:.2f} "
                                f"{'exceeds' if block_route == 'L2' else 'is under'} $2,500")
                case_reason = "R2"
            else:
                block_reason = (f"High-risk signal (p={p:.2f}) with amount far above the "
                                f"card's typical purchase; exposure ${exposure:.2f}")
                case_reason = "fraud episode confirmed by evidence fusion"
            final.append({"action": "BLOCK_CARD", "route": block_route,
                          "reason": block_reason})
            final.append({"action": "CREATE_CASE", "route": "auto", "reason": case_reason})
            if exposure > 1000.0 or shared_origin:
                if shared_origin:
                    report_reason = ("R6: shared device profile links another card; "
                                     "section 3a filing threshold met")
                else:
                    report_reason = (f"exposure ${exposure:.2f} exceeds $1,000; "
                                     "section 3a filing threshold met")
                final.append({"action": "FILE_REPORT", "route": "L2",
                              "reason": report_reason})
            if connected_cards and shared_origin:
                final.append({"action": "MONITOR_CONNECTED_CARDS", "route": "auto",
                              "reason": "R6: cards sharing the same device profile placed "
                                        "under monitoring"})
        else:  # uncertain
            final.append({"action": "MONITOR_CARD", "route": "auto",
                          "reason": "R4: inconclusive evidence within the response window; "
                                    "keep the card active under heightened monitoring"})
            if exposure > 500.0:
                final.append({"action": "ESCALATE_TO_ANALYST", "route": "auto",
                              "reason": f"R8: uncertain verdict with exposure ${exposure:.2f} > $500"})

        files_sar = any(a["action"] == "FILE_REPORT" for a in final)
        self._q(metrics, "action:recommend_nba", 150)
        self._q(metrics, "query:write_case_to_graph", 95)

        # Assemble the answer-file body.
        evidence: List[Dict[str, Any]] = [
            {
                "claim": f"Flagged transaction {flagged} (${amount:.2f}, "
                         f"{'online' if online else 'card-present'}) scored "
                         f"{p0:.2f} by the real-time model"
                         if p0 is not None else
                         f"Customer {ctx.customer_id} disputed transaction {flagged} (${amount:.2f})",
                "source": "graph",
                "ref": f"query:txn_detail(txn_id={flagged})",
                "entity_ids": [flagged],
            }
        ]
        if amount_outlier:
            evidence.append({
                "claim": f"Amount ${amount:.2f} is far above this card's typical "
                         f"purchase (${typical:.2f} median over {len(history_amts)} transactions)",
                "source": "graph",
                "ref": f"query:card_window(card_id={ctx.card_id}, hours=720)",
                "entity_ids": [flagged],
            })
        if device:
            evidence.append({
                "claim": f"Device profile '{device}' links this card to prior "
                         f"activity" + (f" and to closed case(s) {', '.join(similar)}" if similar else ""),
                "source": "graph",
                "ref": f"query:device_neighbors(device_profile={device[:40]})",
                "entity_ids": [c for c in connected_cards] + similar,
            })
        if reply:
            evidence.append({
                "claim": "Customer response to validation request",
                "source": "customer",
                "ref": "evidence_request:1",
                "entity_ids": [],
            })

        status = {
            "fraud": "closed_fraud",
            "legitimate": "closed_legitimate",
            "uncertain": "escalated" if exposure > 500.0 else "open",
        }[verdict]

        summary = (
            f"Case {ctx.case_id} opened on {ctx.trigger_type.replace('_', ' ')} for "
            f"transaction {flagged} (${amount:.2f}) on card {ctx.card_id}. "
            f"Verdict: {verdict} (p={p:.2f}). "
            + ("Customer response drove the verdict. " if reply else "")
            + (f"Shared device profile links {len(connected_cards)} other card(s). "
               if connected_cards else "")
            + f"Pattern: {pattern}."
        )

        sar = {"file": False, "reason": "3a: filing threshold not met; case only, no report",
               "narrative": "", "subjects": [], "total_amount_usd": 0, "activity_dates": []}
        if files_sar:
            subjects = [ctx.customer_id, ctx.card_id] + connected_cards
            if shared_origin:
                sar_rule = "R6"
            elif customer_denied:
                sar_rule = "R2"
            else:
                sar_rule = "3a"
            sar = {
                "file": True,
                "reason": f"{sar_rule}: meets the section 3a filing threshold "
                          f"(exposure or shared-origin linkage)",
                "narrative": (
                    f"On {date}, transaction {flagged} for ${amount:.2f} was "
                    f"{'initiated online' if online else 'made'} using card {ctx.card_id} "
                    f"belonging to customer {ctx.customer_id}. "
                    f"The cardholder stated they did not make this purchase. "
                    + (f"The transaction originated from device profile '{device}', "
                       f"which is also linked to card(s) {', '.join(connected_cards)}. "
                       if device and connected_cards else "")
                    + f"The activity is consistent with {pattern.replace('_', ' ')}. "
                    f"Total suspicious amount: ${exposure:.2f}. "
                    f"The card was blocked and scheduled for reissue; connected cards "
                    f"were placed under monitoring."
                ),
                "subjects": subjects,
                "total_amount_usd": exposure,
                "activity_dates": [date, date],
            }

        if reply:
            what_changed = (
                f"Customer {'denied' if reply.get('denies_transaction') else 'confirmed'} "
                f"the transaction; probability moved from {p0:.2f} to {p:.2f} and the "
                f"recommendation moved from verification to "
                f"{final[0]['action'].replace('_', ' ').lower()}."
            )
        else:
            what_changed = "nothing"
            initial = final  # README: if you requested nothing, final equals initial

        stop_reason = (
            "Customer response settled the verdict; further steps would not change "
            "the actions." if reply else
            f"Fraud probability {p:.2f} crossed the 0.85/0.15 stop band with "
            f"{len(evidence)} evidence items; further steps would not change the decision."
        )

        answer = {
            "case_id": ctx.case_id,
            "case": {
                "status": status,
                "verdict": verdict,
                "fraud_probability": p,
                "pattern": pattern,
                "pattern_description": pattern_desc,
                "affected_txn_ids": affected,
                "first_suspicious_txn_id": flagged if affected else "",
                "connected_card_ids": connected_cards,
                "connected_device_profiles": [device] if device and verdict != "legitimate" else [],
                "exposure_usd": exposure,
                "evidence": evidence,
                "similar_prior_cases": similar,
                "summary": summary,
                "written_to_graph": True,
                "graph_case_id": f"CASE-2016-{int(ctx.case_id.split('-')[1]) + 1200}",
            },
            "evidence_requests": evidence_requests,
            "next_best_actions": {"initial": initial, "final": final,
                                  "what_changed": what_changed},
            "sar": sar,
            "stop_reason": stop_reason,
            "tool_calls": 0,   # filled by the harness from metrics
            "tokens": 0,
            "latency_s": 0.0,
        }

        case_state = {
            "single_signal": single_signal,
            "fraud_probability": p,
            "fraud_probability_initial": p0,
            "customer_denied": customer_denied,
            "customer_confirmed": customer_confirmed,
            "shared_origin": shared_origin,
            "exposure_usd": exposure,
            "verdict": verdict,
            "pattern": pattern,
            "no_reply_24h": verdict == "uncertain",
        }
        # Deterministic simulated 'thinking' so latency_s is plausible.
        time.sleep(0.001 + rng.random() * 0.004)
        return AgentResult(answer=answer, case_state=case_state)
