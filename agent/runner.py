"""GraphAgentRunner: the real investigation loop (spec §6).

LangGraph state machine mirroring the mandated flow:
  trigger -> open_case -> investigate -> assess_uncertainty -> [gate]
      -> (gather_more_evidence -> assess_uncertainty | propose_actions
          -> approval_gate -> execute -> explain -> update_memory -> close)

Implements the bench.agent.AgentRunner protocol, so bench/run.py can run the
stub (deterministic fallback) or this loop interchangeably.

Owns:
  - the single calibrated p (agent.calibrator; the LLM never emits p)
  - the EVSI gather-vs-act gate + computed stopping (agent.gate)
  - policy-as-code routes (actions_api.policy + agent/policy.yaml)
  - structured what_changed, 6-section summary, 8-sentence SAR scaffold
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

from actions_api import policy
from bench.agent import AgentResult, CaseContext
from bench.dataset import DatasetIndex, norm_txn_id
from bench.instrument import CaseMetrics

from agent.decision_model import DecisionModel, MODEL_PATH
from agent.gate import decide as gate_decide, load_costs
from agent.llm import EchoProvider, LLMProvider
from agent.tools import InvestigationTools

class AgentState(TypedDict, total=False):
    ctx: Dict[str, Any]
    idx: Any
    metrics: Any
    tools: Any
    cal: Any
    evidence: List[Dict[str, Any]]
    requests: List[Dict[str, Any]]
    similar: List[str]
    connected_cards: List[str]
    pattern: str
    pattern_desc: str
    scores: Dict[str, Any]
    p: float
    p_lo: float
    p_hi: float
    p_init: float
    p_before_round: float
    round_n: int
    gathered: List[str]
    stop_reason: str
    steps: List[Dict[str, Any]]
    replies: List[Dict[str, Any]]
    final_action: str
    verdict: str
    act: str
    gather_type: str
    pending_evidence: str
    skip_cal: bool
    evsi_table: Dict[str, float]


class GraphAgentRunner:
    """The LangGraph loop behind the bench protocol."""

    def __init__(self, persona: Any = None, seed: int = 42,
                 llm: Optional[LLMProvider] = None, mode: str = "stub",
                 policy_path: Optional[Path] = None,
                 calibrator_path: Path = MODEL_PATH,
                 memory_mode: str = "full",
                 skip_calibration: bool = False,
                 neutralize_graph: bool = False) -> None:
        self.memory_mode = memory_mode
        self.skip_calibration = skip_calibration
        self.neutralize_graph = neutralize_graph
        self.seed = seed
        self.persona = persona
        self.llm = llm or EchoProvider()
        self.mode = mode
        self.costs = load_costs(policy_path)
        self.calibrator_path = calibrator_path
        self._cal: Optional[DecisionModel] = None
        self._build_graph()

    # -- calibrator -------------------------------------------------------
    def _calibrator(self, idx: DatasetIndex) -> DecisionModel:
        if self._cal is None:
            self._cal = DecisionModel(self.calibrator_path)
        return self._cal

    # -- LangGraph construction -------------------------------------------
    def _build_graph(self) -> None:
        from langgraph.graph import END, StateGraph

        # state schema: module-level AgentState

        def open_case(s: AgentState) -> AgentState:
            s["evidence"] = []
            s["requests"] = []
            s["round_n"] = 0
            s["gathered"] = []
            s["steps"] = []
            s["replies"] = []
            s["stop_reason"] = ""
            s["pending_evidence"] = ""
            s["steps"].append({"step": "open_case", "case_id": s["ctx"]["case_id"]})
            return s

        def investigate(s: AgentState) -> AgentState:
            ctx, idx, tools = s["ctx"], s["idx"], s["tools"]
            t = norm_txn_id(ctx["flagged_txn_id"])
            pack = tools.q_evidence_pack(ctx["case_id"], ctx["flagged_txn_id"], ctx["card_id"])
            ev = pack.data.get("flagged_txn", {})
            s["evidence"].append({
                "claim": (f"Flagged transaction {t} (${ev.get('amount_usd', 0):.2f}, "
                          f"{'online' if ev.get('online') else 'card-present'})"),
                "source": "graph", "ref": "q_evidence_pack", "entity_ids": [t],
            })
            sims = tools.q_case_structural_similarity(ev.get("device", ""),
                                                      case_id=ctx["case_id"])
            s["similar"] = sims.data.get("similar_prior_cases", [])
            device = ev.get("device", "") or idx.txn_device.get(t, "")
            device_cards = ({idx.txn_card.get(tx)
                             for tx, profile in idx.txn_device.items()
                             if profile == device} - {None}) if device else set()
            s["connected_cards"] = sorted({
                c for cc in s["similar"]
                if 2 <= len(device_cards) <= 5
                if cc in idx.closed_case_outcome_fraud
                if device and any(idx.txn_device.get(tx) == device
                                  for tx in idx.closed_case_txns.get(cc, ()))
                for c in idx.closed_case_cards.get(cc, ())
                if c != ctx["card_id"] and idx.card_exists(c)
            })[:3]
            if s["similar"]:
                s["evidence"].append({
                    "claim": f"Device/structure links prior case(s) {', '.join(s['similar'])}",
                    "source": "graph", "ref": "q_case_structural_similarity",
                    "entity_ids": s["similar"],
                })
            s["scores"] = {
                "card_testing": tools.q_card_testing_score(
                    ctx["card_id"], ctx["flagged_txn_id"]
                ).data,
                "new_device_cnp": tools.q_new_device_cnp_score(ctx["flagged_txn_id"]).data,
                "out_of_region": tools.q_out_of_region_score(ctx["flagged_txn_id"]).data,
                "ato": tools.q_ato_score(ctx["customer_id"]).data,
                "recurring": tools.q_recurring_entities(
                    ctx["card_id"], ctx["flagged_txn_id"]
                ).data,
            }
            device = ev.get("device", "") or idx.txn_device.get(
                norm_txn_id(ctx["flagged_txn_id"]), "")
            s["pattern"], s["pattern_desc"] = self._classify(s["scores"], device=device)
            if s["pattern"] == "undocumented":
                from agent.discovery import match_case_device
                exemplar = (match_case_device(device) or {}).get("exemplar", {})
                ring_cases = [
                    cc for cc in exemplar.get("case_ids", [])
                    if cc in idx.closed_case_outcome_fraud
                    and any(idx.txn_device.get(tx) == device
                            for tx in idx.closed_case_txns.get(cc, ()))
                ]
                s["similar"] = list(dict.fromkeys(ring_cases + s["similar"]))
                s["connected_cards"] = sorted(set(s["connected_cards"]) | {
                    card for cc in ring_cases
                    for card in idx.closed_case_cards.get(cc, ())
                    if card != ctx["card_id"] and idx.card_exists(card)
                })[:5]
            # Policy grounding (GraphRAG): retrieve the governing policy text for
            # the classified pattern and record it as citable document evidence.
            try:
                from graphrag.ingest import retrieve as _doc_retrieve
                _pol = _doc_retrieve(f"{s['pattern']} fraud policy SAR filing threshold", k=1)
                if _pol:
                    _d = _pol[0]
                    s["evidence"].append({
                        "claim": (f"Policy check ({_d['source']}#{_d['heading']}): "
                                  f"{_d['text'][:160].strip()}"),
                        "source": "document", "ref": "graphrag.retrieve",
                        "entity_ids": [],
                    })
            except Exception:
                pass  # document retrieval is additive; never blocks the loop
            if s["pattern"] == "undocumented":
                s["evidence"].append({
                    "claim": (
                        "Device fingerprint matches a Build/-qualified "
                        "cross-account shared-device ring "
                        "(permutation-tested; see eval/pattern_discovery.json)"
                    ),
                    "source": "graph",
                    "ref": "pattern_discovery:cross_account_device_ring",
                    "entity_ids": [device] if device else [],
                })
            s["steps"].append({"step": "investigate", "tools": sorted(s["scores"].keys())})
            return s

        def assess(s: AgentState) -> AgentState:
            ctx, tools, cal = s["ctx"], s["tools"], s["cal"]
            if s.get("skip_cal"):
                # no-calibration ablation: p = trigger prior, no CI update.
                from agent.calibrator import CalibratedP
                prior = float(ctx.get("risk_score") or 0.45)
                cp = CalibratedP(p=prior, lo=prior, hi=prior)
            else:
                feats = tools.q_case_feature_vector(
                    ctx["flagged_txn_id"], ctx["card_id"], ctx["customer_id"],
                    prior_score=ctx.get("risk_score"),
                ).data["features"]
                cp = cal.predict_ci(feats, ctx["flagged_txn_id"])
            # Customer stance: documented Bayesian update on the calibrated p
            # (single-p invariant; most recent reply wins).
            from agent.calibrator import update_with_stance
            stance = "silent"
            if s["replies"]:
                last = s["replies"][-1]
                stance = ("denied" if last.get("denied")
                          else ("confirmed" if last.get("confirmed") else "silent"))
                p_upd = update_with_stance(cp.p, stance)
                # CI carries over (base-evidence uncertainty), clipped around p.
                lo = min(cp.lo, p_upd)
                hi = max(cp.hi, p_upd)
                cp = type(cp)(p=p_upd, lo=lo, hi=hi)
            s["p_before_round"] = s.get("p", cp.p)
            s["p"], s["p_lo"], s["p_hi"] = cp.p, cp.lo, cp.hi
            s["steps"].append({"step": "assess", "p": round(cp.p, 3),
                               "ci": [round(cp.lo, 3), round(cp.hi, 3)]})
            return s

        def gate(s: AgentState) -> AgentState:
            t = norm_txn_id(s["ctx"]["flagged_txn_id"])
            exposure = abs(s["idx"].amount(t))
            evidence = s["evidence"]
            if s["round_n"] > 0 and (s["p"] - 0.5) * (s["p_before_round"] - 0.5) < 0:
                s["stop_reason"] = "response_flip:p_crossed_0.5_after_new_evidence"
                s["final_action"] = ""
                s["act"] = "act"
                return s
            d = gate_decide(s["p"], (s["p_lo"], s["p_hi"]), exposure, self.costs,
                            evidence, s["round_n"], max_rounds=3,
                            gathered=s["gathered"])
            s["evsi_table"] = d.evsi_table
            if d.gather:
                s["gather_type"] = d.action
                s["act"] = "gather"
            else:
                s["final_action"] = d.action
                s["stop_reason"] = d.stop_reason or s.get("stop_reason", "")
                s["act"] = "act"
            s["steps"].append({"step": "gate", "gather": d.gather,
                               "action": d.action, "stop": d.stop_reason})
            return s

        def gather(s: AgentState) -> AgentState:
            g = s.get("gather_type", "customer_validation")
            ctx = s["ctx"]
            t = norm_txn_id(ctx["flagged_txn_id"])
            amount = s["idx"].amount(t)
            s["gathered"].append(g)
            s["round_n"] += 1
            if g == "customer_validation" and self.persona is not None:
                reply = s["tools"].verify_with_customer(
                    ctx["case_id"],
                    f"Did you make transaction {t} (${amount:.2f})?",
                    persona=self.persona,
                    context={"txn_id": t, "amount": amount},
                ).data
                denied = bool(reply.get("denies_transaction"))
                confirmed = bool(reply.get("confirmed_transaction"))
                s["replies"].append({"denied": denied, "confirmed": confirmed,
                                     "silent": not denied and not confirmed})
                s["requests"].append({"type": g, "asked_after_step": 2 * s["round_n"],
                                      "assumed_response": reply.get("response", "")})
                stance = ("denied" if denied else ("confirmed" if confirmed else "no reply within window"))
                s["evidence"].append({"claim": f"Customer {stance} the transaction (round {s['round_n']})",
                                      "source": "customer", "ref": f"evidence_request:{s['round_n']}",
                                      "entity_ids": []})
            else:
                s["pending_evidence"] = g
                s["final_action"] = {
                    "customer_validation": "VERIFY_WITH_CUSTOMER",
                    "step_up_auth": "STEP_UP_AUTH",
                    "analyst_info": "REQUEST_ANALYST_INFO",
                }[g]
                s["stop_reason"] = f"awaiting_evidence_response:{g}"
                s["requests"].append({"type": g, "asked_after_step": 2 * s["round_n"],
                                      "assumed_response": "No response supplied; request pending"})
            s["steps"].append({"step": "gather", "type": g,
                               "status": "pending" if s["pending_evidence"] else "answered"})
            return s

        def propose(s: AgentState) -> AgentState:
            s["steps"].append({"step": "propose", "action": s["final_action"]})
            return s

        def approval(s: AgentState) -> AgentState:
            # The prototype records required routing; it cannot approve
            # a bank action on behalf of a human reviewer.
            selected = str(s.get("final_action", ""))
            exposure = abs(s["idx"].amount(s["ctx"]["flagged_txn_id"]))
            try:
                route = policy.required_route(policy.ActionName(selected), exposure).value
            except (ValueError, KeyError):
                route = "auto"
            s["steps"].append({"step": "approval", "route": route,
                               "approved": route == "auto", "simulated": True})
            return s

        def execute(s: AgentState) -> AgentState:
            s["steps"].append({"step": "execute", "action": s["final_action"],
                               "simulated": True})
            return s

        def explain(s: AgentState) -> AgentState:
            # LLM slot-fill happens in _summary/_sar; here we just record it.
            s["steps"].append({"step": "explain", "llm": type(self.llm).__name__})
            return s

        def update_memory(s: AgentState) -> AgentState:
            s["steps"].append({"step": "update_memory",
                               "similar": len(s.get("similar", []))})
            return s

        def close(s: AgentState) -> AgentState:
            s["steps"].append({"step": "close"})
            return s

        g = StateGraph(AgentState)
        g.add_node("open_case", open_case)
        g.add_node("investigate", investigate)
        g.add_node("assess_uncertainty", assess)
        g.add_node("gate", gate)
        g.add_node("gather_more_evidence", gather)
        g.add_node("propose_actions", propose)
        g.add_node("approval_gate", approval)
        g.add_node("execute", execute)
        g.add_node("explain", explain)
        g.add_node("update_memory", update_memory)
        g.add_node("close", close)
        g.set_entry_point("open_case")
        g.add_edge("open_case", "investigate")
        g.add_edge("investigate", "assess_uncertainty")
        g.add_edge("assess_uncertainty", "gate")

        def route(s: AgentState) -> str:
            return "gather_more_evidence" if s.get("act") == "gather" else "propose_actions"

        g.add_conditional_edges("gate", route,
                                {"gather_more_evidence": "gather_more_evidence",
                                 "propose_actions": "propose_actions"})
        g.add_conditional_edges("gather_more_evidence",
                                lambda s: "propose_actions" if s.get("pending_evidence") else "assess_uncertainty",
                                {"propose_actions": "propose_actions",
                                 "assess_uncertainty": "assess_uncertainty"})
        g.add_edge("propose_actions", "approval_gate")
        g.add_edge("approval_gate", "execute")
        g.add_edge("execute", "explain")
        g.add_edge("explain", "update_memory")
        g.add_edge("update_memory", "close")
        g.add_edge("close", END)
        self._graph = g.compile()

    # -- pattern classification from scorer outputs ------------------------
    def _classify(self, scores: Dict[str, Any], device: str = "") -> tuple:
        # Undocumented sixth typology wins when discovery shipped a ring and
        # this case's device is in it (permutation p<0.05 gate already passed).
        from agent.discovery import match_case_device
        hit = match_case_device(device)
        if hit is not None:
            return "undocumented", hit["pattern_description"]
        ct = scores.get("card_testing", {})
        nd = scores.get("new_device_cnp", {})
        oor = scores.get("out_of_region", {})
        if (ct.get("small_auths", 0) >= 3
                and ct.get("probe_to_large", 0) > 0
                and ct.get("score", 0) >= 0.5):
            return "card_testing", ""
        if (nd.get("new_device") and nd.get("cnp")
                and nd.get("score", 0) >= 0.85
                and nd.get("contradict_count", 0) == 0
                and (nd.get("w_email_mismatch", 0) > 0
                     or nd.get("w_amount_outlier", 0) > 0)):
            return "card_not_present_new_device", ""
        if not nd.get("cnp") and oor.get("score", 0) >= 0.5:
            return "out_of_region_use", ""
        return "none", ""

    # -- the protocol entry point ------------------------------------------
    def investigate(self, ctx: CaseContext, idx: DatasetIndex,
                    metrics: CaseMetrics,
                    on_step: Optional[Callable[[Dict[str, Any]], None]] = None) -> AgentResult:
        cal = self._calibrator(idx)
        tools = InvestigationTools(idx, metrics=metrics, mode=self.mode,
                                   memory_mode=self.memory_mode,
                                   neutralize_graph=self.neutralize_graph)
        state = AgentState(
            ctx={"case_id": ctx.case_id, "opened_at": ctx.opened_at,
                 "trigger_type": ctx.trigger_type, "trigger_text": ctx.trigger_text,
                 "flagged_txn_id": ctx.flagged_txn_id, "card_id": ctx.card_id,
                 "customer_id": ctx.customer_id, "risk_score": ctx.risk_score},
            idx=idx, metrics=metrics, tools=tools, cal=cal,
            evidence=[], requests=[], replies=[], similar=[], connected_cards=[],
            pattern="none", pattern_desc="", scores={}, p=0.0, p_lo=0.0,
            p_hi=1.0, p_init=0.0, p_before_round=0.0, round_n=0, gathered=[],
            stop_reason="", steps=[], final_action="", verdict="", act="",
            gather_type="", pending_evidence="", skip_cal=self.skip_calibration,
            evsi_table={},
        )
        if on_step is None:
            state = self._graph.invoke(state, {"recursion_limit": 40})
        else:
            seen = 0
            for update in self._graph.stream(state, {"recursion_limit": 40}, stream_mode="values"):
                steps = update.get("steps", [])
                for step in steps[seen:]:
                    on_step(step)
                seen = len(steps)
                state = update
        return self._build_answer(ctx, idx, metrics, state)

    # -- answer assembly (contract = bench/schema.py) -----------------------
    def _build_answer(self, ctx: CaseContext, idx: DatasetIndex, metrics: CaseMetrics,
                      s: Dict[str, Any]) -> AgentResult:
        t = norm_txn_id(ctx.flagged_txn_id)
        amount = idx.amount(t)
        date = idx.date_of(t)
        device = idx.txn_device.get(t, "")
        online = idx.txn_online.get(t, True)
        p = round(float(s["p"]), 2)
        p0 = round(float(s.get("p_before_round") or s["p"]), 2)
        verdict = "fraud" if p >= 0.85 else ("legitimate" if p <= 0.15 else "uncertain")
        exposure = round(abs(amount), 2) if verdict != "legitimate" else 0.0
        affected = [t] if verdict != "legitimate" else []
        similar = list(s.get("similar", []))
        connected_cards = list(s.get("connected_cards", []))
        linked_fraud_cases = [
            cc for cc in similar
            if connected_cards
            if cc in idx.closed_case_outcome_fraud
            and device
            and any(idx.txn_device.get(tx) == device
                    for tx in idx.closed_case_txns.get(cc, ()))
        ]

        # Rule-fired computation FIRST; citations filtered through it.
        single_signal = ctx.trigger_type in ("risk_score", "analyst_request")
        customer_denied = any(r.get("denied") for r in s["replies"])
        customer_confirmed = any(r.get("confirmed") for r in s["replies"])
        no_reply = any(r.get("silent") for r in s["replies"])
        testing_score = s["scores"].get("card_testing", {})
        card_testing_sequence = (testing_score.get("small_auths", 0) >= 3
                                 and testing_score.get("probe_to_large", 0) > 0)
        shared_origin = bool(device and connected_cards)
        recurring = s["scores"].get("recurring", {}).get("recurring", [])
        matches_recurring = bool(device and device in recurring)
        conflicting_evidence = customer_denied and p < 0.5
        pattern = s["pattern"]
        coordinated_abuse = pattern == "undocumented"
        case_state = {
            "single_signal": single_signal,
            "fraud_probability": p,
            "fraud_probability_initial": p0,
            "customer_denied": customer_denied,
            "customer_confirmed": customer_confirmed,
            "no_reply_24h": no_reply,
            "card_testing_sequence": card_testing_sequence,
            "shared_origin": shared_origin,
            "matches_recurring_pattern": matches_recurring,
            "conflicting_evidence": conflicting_evidence,
            "exposure_usd": exposure,
            "verdict": verdict,
            "pattern": pattern,
            "coordinated_abuse": coordinated_abuse,
        }
        fired = set(policy.fired_rules(case_state))

        def cite(rule: str, text: str) -> str:
            return f"{rule}: {text}" if rule in fired else text

        # Initial NBA (before any evidence): gate arithmetic on p0.
        initial: List[Dict[str, str]] = []
        if single_signal and min(p0, p) < 0.70:
            initial.append({"action": "VERIFY_WITH_CUSTOMER", "route": "auto",
                            "reason": cite("R1", f"single weak signal (p={p0:.2f}) - verify before any block")})
        elif p0 >= 0.85:
            initial.append({"action": "DECLINE_TRANSACTION", "route": "L1",
                            "reason": f"risk score {p0:.2f} >= 0.85 on the flagged authorization"})
        else:
            initial.append({"action": "MONITOR_CARD", "route": "auto",
                            "reason": f"ambiguous signal (p={p0:.2f}); monitor while gathering evidence"})

        # Final NBAs from verdict + gate.
        final: List[Dict[str, str]] = []
        if verdict == "legitimate":
            final.append({"action": "CLOSE_NO_FRAUD", "route": "auto",
                          "reason": cite("R3", "evidence supports the legitimate hypothesis")})
        elif verdict == "fraud":
            if p0 < 0.85:
                final.append({"action": "DECLINE_TRANSACTION", "route": "L1",
                              "reason": f"post-evidence p={p:.2f}; decline the pending authorization"})
            block_route = "L2" if exposure > policy.BLOCK_CARD_L2_THRESHOLD_USD else "L1"
            reason = cite("R2", "customer denial") if customer_denied else \
                f"p={p:.2f} with {len(s['evidence'])} evidence items"
            final.append({"action": "BLOCK_CARD", "route": block_route,
                          "reason": f"{reason}; exposure ${exposure:.2f} "
                                    f"{'exceeds' if block_route == 'L2' else 'is under'} $2,500"})
            final.append({"action": "CREATE_CASE", "route": "auto", "reason": "fraud episode confirmed"})
            files_sar = exposure > 1000.0 or shared_origin or coordinated_abuse
            if files_sar:
                if "R9" in fired:
                    fr = cite("R9", "validated cross-account ring; section 3a filing threshold met")
                elif "R6" in fired:
                    fr = cite("R6", "shared device profile links another card; section 3a threshold met")
                elif "R5" in fired:
                    fr = cite("R5", "card-testing sequence; section 3a threshold met")
                elif "R2" in fired:
                    fr = cite("R2", "confirmed fraudulent transaction; section 3a threshold met")
                else:
                    fr = f"exposure ${exposure:.2f} exceeds $1,000; section 3a filing threshold met"
                final.append({"action": "FILE_REPORT", "route": "L2", "reason": fr})
            if shared_origin and connected_cards:
                final.append({"action": "MONITOR_CONNECTED_CARDS", "route": "auto",
                              "reason": cite("R6", "cards sharing the device profile placed under monitoring")})
            if "R9" in fired:
                final.append({"action": "ESCALATE_TO_ANALYST", "route": "auto",
                              "reason": cite("R9", "review the undocumented cross-account pattern")})
        else:  # uncertain
            final.append({"action": "MONITOR_CARD", "route": "auto",
                          "reason": cite("R4", "monitor while the requested evidence is pending"
                                         if s.get("pending_evidence") else
                                         "inconclusive within the response window; heightened monitoring")})
            if exposure > 500.0:
                final.append({"action": "ESCALATE_TO_ANALYST", "route": "auto",
                              "reason": cite("R8", f"uncertain verdict with exposure ${exposure:.2f} > $500")})
        # The expected-cost gate is authoritative for the first action. Keep
        # policy-derived follow-up actions, but never publish a different
        # primary action than the one the loop actually chose.
        selected = str(s.get("final_action", ""))
        if selected and selected not in {a["action"] for a in final}:
            try:
                route = policy.required_route(policy.ActionName(selected), exposure).value
                reason = (f"evidence request has positive expected value "
                          f"(${max(s.get('evsi_table', {}).values(), default=0):.2f}); "
                          "fraud decision remains pending"
                          if s.get("pending_evidence") else
                          "monitor immediately while the L1 card block awaits approval"
                          if selected == "MONITOR_CARD" and verdict == "fraud" else
                          f"decision gate selected {selected} as the minimum expected-cost action")
                final.insert(0, {"action": selected, "route": route,
                                 "reason": reason})
            except (ValueError, KeyError):
                pass
        files_sar = any(a["action"] == "FILE_REPORT" for a in final)
        # README: if nothing was requested, the final recommendation equals the initial one.
        if not s["requests"]:
            initial = [dict(a) for a in final]

        # Evidence + requests from the loop.
        evidence = list(s["evidence"])
        requests = list(s["requests"])

        # SAR (deterministic 8-sentence scaffold; LLM slot-fill only).
        sar = {"file": False, "reason": "3a: filing threshold not met; case only, no report",
               "narrative": "", "subjects": [], "total_amount_usd": 0, "activity_dates": []}
        if files_sar:
            subjects = [ctx.customer_id, ctx.card_id] + connected_cards
            sar = {
                "file": True,
                "reason": next((a["reason"] for a in final if a["action"] == "FILE_REPORT"), ""),
                "narrative": (
                    f"On {date}, transaction {t} for ${amount:.2f} was "
                    f"{'initiated online' if online else 'made'} using card {ctx.card_id} "
                    f"belonging to customer {ctx.customer_id}. "
                    f"The transaction was flagged by the {ctx.trigger_type.replace('_', ' ')} trigger. "
                    f"Graph evidence places the transaction on device profile "
                    f"'{device[:40] if device else 'unknown'}'. "
                    + ("The cardholder was asked to validate the charge and denied it. "
                       if customer_denied else "")
                    + f"The assessed fraud probability is {p:.2f}. "
                    + (f"The same device profile is linked to confirmed-fraud case(s) {', '.join(linked_fraud_cases)}. "
                       if linked_fraud_cases else "")
                    + f"Total suspicious amount: ${exposure:.2f}. "
                    f"Blocking the card was recommended pending reissue."
                    + (" Monitoring the connected cards was also recommended."
                       if connected_cards else "")
                ),
                "subjects": subjects,
                "total_amount_usd": exposure,
                "activity_dates": [date, date],
            }

        # Structured what_changed (JSON string inside the schema's string type).
        if s.get("pending_evidence"):
            what_changed = (f"Requested {s['final_action']} before deciding; "
                            "no response was supplied, so the fraud decision remains pending.")
        elif requests:
            what_changed = json.dumps([{
                "evidence_id": f"evidence_request:{i + 1}",
                "rule_id": "R3" if "R3" in fired else ("R2" if "R2" in fired else "R4"),
                "added": r["type"],
                "removed": "VERIFY_WITH_CUSTOMER" if i == 0 else "",
                "p_before": round(p0, 2),
                "p_after": round(p, 2),
            } for i, r in enumerate(requests)])
        else:
            what_changed = "nothing"

        status = {"fraud": "closed_fraud", "legitimate": "closed_legitimate",
                  "uncertain": "escalated" if exposure > 500.0 else "open"}[verdict]

        def _stop_human(sr: str) -> str:
            """Machine stop code -> one plain-sentence clause for the summary."""
            if sr.startswith("evsi_le_0"):
                act = sr.split(":")[1].split("_optimal")[0].replace("_", " ").lower() if ":" in sr else "the recommended action"
                return f"further evidence has no expected value, so the loop stopped and recommended to {act}"
            if sr.startswith("robust"):
                import re as _re
                m = _re.search(r":([A-Z_]+)", sr)
                act = m.group(1).strip("_").replace("_", " ").lower() if m else "the recommended action"
                return f"{act} remains cost-optimal for every p in the 90% CI, so extra evidence cannot flip the decision"
            if sr.startswith("p_extreme"):
                return "two independent evidence classes agree and p is outside the ambiguous band"
            if sr.startswith("awaiting_evidence_response"):
                return "an evidence request is pending; no customer or analyst response was invented"
            return sr.replace("_", " ").lower()

        summary = (
            f"Trigger: {ctx.trigger_type.replace('_', ' ')} on transaction {t} "
            f"(${amount:.2f}, card {ctx.card_id}). "
            f"Findings: p={p:.2f} (90% CI {s['p_lo']:.2f}-{s['p_hi']:.2f}), "
            f"{len(evidence)} evidence items. "
            f"Pattern: {pattern}. "
            f"Stopping: {_stop_human(s['stop_reason'] or 'robust-optimal:' + s['final_action'] + '_for_all_p_in_ci')}. "
            f"Actions: {' -> '.join(a['action'] for a in final)}; approval in-line. "
            f"Memory: {len(similar)} closed prior case(s) cited."
        )
        stop_reason = s["stop_reason"] or (
            f"robust-optimal: {s['final_action']} cost-optimal across the 90% CI"
        )

        # Optional real-trace dump for the UI timeline (env: SUMORA_TRACE_DIR).
        import os as _os
        trace_dir = _os.environ.get("SUMORA_TRACE_DIR")
        if trace_dir:
            from pathlib import Path as _P
            import json as _json
            _td = _P(trace_dir)
            _td.mkdir(parents=True, exist_ok=True)
            (_td / f"{ctx.case_id}.trace.json").write_text(_json.dumps({
                "case_id": ctx.case_id,
                "steps": s["steps"],
                "tool_log": list(metrics.tool_log),
                "mcp_calls": s["tools"].mcp_calls,
                "p_final": p, "p_initial": p0,
                "evsi_table": s.get("evsi_table", {}),
                "stop_reason": stop_reason,
                "evidence": evidence,
                "requests": requests,
                "similar_prior_cases": similar,
            }, indent=2))

        # Mandated capability: the case returns to the graph. Live mode writes
        # the decision + audit trail via q_record_case and friends; stub mode
        # reports written_to_graph=false honestly.
        written = s["tools"].record_case(
            case_id=ctx.case_id, status=status, verdict=verdict, p=p,
            pattern=pattern, pattern_desc=s["pattern_desc"],
            affected_txn_ids=affected, exposure_usd=exposure,
            summary=summary, stop_reason=stop_reason,
            similar_case_ids=similar, evidence=evidence,
            initial_actions=initial, final_actions=final,
            what_changed=what_changed, steps=s["steps"])

        answer = {
            "case_id": ctx.case_id,
            "case": {
                "status": status,
                "verdict": verdict,
                "fraud_probability": p,
                "pattern": pattern,
                "pattern_description": s["pattern_desc"],
                "affected_txn_ids": affected,
                "first_suspicious_txn_id": t if affected else "",
                "connected_card_ids": connected_cards,
                "connected_device_profiles": [device] if device and verdict != "legitimate" else [],
                "exposure_usd": exposure,
                "evidence": evidence,
                "similar_prior_cases": similar,
                "summary": summary,
                "written_to_graph": written,
                "graph_case_id": ctx.case_id if written else "",
            },
            "evidence_requests": requests,
            "next_best_actions": {"initial": initial, "final": final,
                                  "what_changed": what_changed},
            "sar": sar,
            "stop_reason": stop_reason,
            "tool_calls": 0,
            "tokens": 0,
            "latency_s": 0.0,
        }
        return AgentResult(answer=answer, case_state=case_state)
