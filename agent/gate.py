"""Decision layer (spec §6): expected-cost action choice, EVSI gather-vs-act
gate, and computed stopping.

Everything here is deterministic arithmetic over the calibrated p (and its
bootstrap CI) - the LLM never makes these trade-offs, it only narrates them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def load_costs(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else Path(__file__).resolve().parent / "policy.yaml"
    return yaml.safe_load(p.read_text())


# ---------------------------------------------------------------------------
# Expected-cost action choice
# ---------------------------------------------------------------------------

@dataclass
class ActionOption:
    action: str
    # Cost if the case is actually fraud (acting wrongly-lenient loses exposure).
    cost_if_fraud: float
    # Cost if the case is actually legitimate (acting wrongly-strict pays friction).
    cost_if_legit: float


@dataclass
class GateDecision:
    action: str
    expected_cost: float
    gather: bool
    evsi_best: float
    evsi_table: Dict[str, float] = field(default_factory=dict)
    stop_reason: str = ""
    ci: Tuple[float, float] = (0.0, 1.0)
    p: float = 0.0


def expected_cost(p: float, exposure: float, opt: ActionOption, costs: Dict[str, Any]) -> float:
    """EC(a) = p * C(a|fraud) + (1-p) * C(a|legit)."""
    return p * opt.cost_if_fraud + (1.0 - p) * opt.cost_if_legit


def build_options(exposure: float, costs: Dict[str, Any]) -> Dict[str, ActionOption]:
    """The action space for the gate. Costs follow agent/policy.yaml."""
    c_fn = costs["costs"]["C_FN_frac"] * exposure
    c_dec = costs["costs"]["C_FP_decline_usd"]
    c_blk = costs["costs"]["C_FP_block_usd"]
    return {
        # Lenient actions: pay full exposure if fraud, ~0 if legit.
        "MONITOR_CARD": ActionOption("MONITOR_CARD", c_fn, 0.0),
        "CLOSE_NO_FRAUD": ActionOption("CLOSE_NO_FRAUD", c_fn, 0.0),
        # Strict actions: exposure averted if fraud, friction if legit.
        "DECLINE_TRANSACTION": ActionOption("DECLINE_TRANSACTION", c_dec, c_dec),
        "BLOCK_CARD": ActionOption("BLOCK_CARD", min(c_blk, c_fn), c_blk),
    }


def choose_action(p: float, exposure: float, costs: Dict[str, Any],
                  allowed: Optional[List[str]] = None) -> Tuple[str, float]:
    """a* = argmin_a EC(a) over the allowed action space."""
    opts = build_options(exposure, costs)
    if allowed:
        opts = {k: v for k, v in opts.items() if k in allowed}
    best = min(opts.values(), key=lambda o: expected_cost(p, exposure, o, costs))
    return best.action, expected_cost(p, exposure, best, costs)


# ---------------------------------------------------------------------------
# EVSI: expected value of sample information for each evidence action
# ---------------------------------------------------------------------------

def _evsi_for_gather(gather_type: str, p: float, exposure: float,
                     costs: Dict[str, Any]) -> float:
    """EVSI(g) = EC_pre(gather) - E_post[EC*(post)] - C_evidence(g).

    Two-outcome Bayesian update with a documented per-type discriminating
    power (True-positive rate / False-positive rate of the evidence signal).
    These TPR/FPR priors are the honest weak-evidence model: a customer
    denial is strong, a silent reply is weak.
    """
    disc = {
        # tpr = P(implicating signal | fraud), fpr = P(implicating signal | legit)
        "customer_validation": (0.75, 0.05),
        "step_up_auth": (0.90, 0.02),
        "analyst_info": (0.60, 0.10),
    }.get(gather_type, (0.5, 0.5))

    c_evidence = costs["costs"]["C_evidence_usd"].get(gather_type, 5.0)
    c_delay = costs["costs"]["C_delay_frac"] * exposure

    tpr, fpr = disc
    # Optimal post-gather action cost under each signal outcome.
    def post_ec(sig: bool) -> float:
        # P(fraud | signal) via Bayes.
        num = p * (tpr if sig else (1 - tpr))
        den = num + (1 - p) * (fpr if sig else (1 - fpr))
        post_p = num / den if den > 0 else p
        _, ec = choose_action(post_p, exposure, costs)
        return ec

    a_pre, ec_pre = choose_action(p, exposure, costs)
    p_sig = p * tpr + (1 - p) * fpr
    e_post = p_sig * post_ec(True) + (1 - p_sig) * post_ec(False)
    return ec_pre - e_post - c_evidence - c_delay


def evsi_table(p: float, exposure: float, costs: Dict[str, Any]) -> Dict[str, float]:
    return {g: round(_evsi_for_gather(g, p, exposure, costs), 2)
            for g in ("customer_validation", "step_up_auth", "analyst_info")}


def best_gather(p: float, exposure: float, costs: Dict[str, Any]) -> Tuple[Optional[str], float]:
    tab = evsi_table(p, exposure, costs)
    g = max(tab, key=tab.get)
    return (g, tab[g]) if tab[g] > costs["thresholds"]["evsi_min_usd"] else (None, tab[g])


# ---------------------------------------------------------------------------
# Computed stopping (spec §6): four machine-checkable rules
# ---------------------------------------------------------------------------

def evidence_class(item: Dict[str, Any]) -> str:
    """Map an evidence item to its class (device / behavioral / network /
    memory / customer-response) - the executable form of 'independent'."""
    haystack = " ".join(str(item.get(k, "")) for k in ("source", "ref", "claim", "type"))
    if "customer" in haystack or item.get("type") in ("customer_validation", "step_up_auth"):
        return "customer-response"
    if "similar" in haystack or "prior_case" in haystack or "memory" in haystack:
        return "memory"
    if "device" in haystack or "region" in haystack:
        return "device"
    if "cluster" in haystack or "email" in haystack or "pattern" in haystack:
        return "network"
    return "behavioral"


def stopping_reason(p: float, ci: Tuple[float, float], exposure: float,
                    costs: Dict[str, Any], evidence: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Return (should_stop, reason). Four rules, first match wins:
      1. robust-optimal: the same action is cost-optimal for every p in the CI
      2. max EVSI <= 0
      3. (reserved for response flips - handled by the loop when a reply flips p)
      4. p >= 0.85 or <= 0.15 with >=2 items from distinct evidence classes
    """
    # Robust optimality alone is not enough to stop. Positive EVSI means a
    # verification step can still change the decision even when the current
    # action is stable across the interval.
    g, v = best_gather(p, exposure, costs)
    acts = {choose_action(q, exposure, costs)[0] for q in ci}
    if len(acts) == 1 and g is None:
        a, _ = choose_action(p, exposure, costs)
        return True, f"robust_optimal:{a}_for_all_p_in_ci"

    # 2. No valuable evidence left to buy.
    if g is None:
        a, _ = choose_action(p, exposure, costs)
        return True, f"evsi_le_0:{a}_optimal_max_evsi_{v:.2f}"

    # 4. Confident p with >=2 distinct evidence classes.
    p_hi = costs["thresholds"]["p_high"]
    p_lo = costs["thresholds"]["p_low"]
    classes = {evidence_class(e) for e in evidence}
    if (p >= p_hi or p <= p_lo) and len(classes) >= 2:
        side = "fraud" if p >= p_hi else "legitimate"
        return True, f"confident_p:{side}_p_{p:.2f}_classes_{len(classes)}"

    return False, ""


def decide(p: float, ci: Tuple[float, float], exposure: float, costs: Dict[str, Any],
           evidence: List[Dict[str, Any]], rounds_done: int, max_rounds: int = 4,
           gathered: Optional[List[str]] = None) -> GateDecision:
    """The gate: gather iff best EVSI > 0 AND the robust-optimal rule has not
    fired; otherwise act. Bounded by max_rounds (graceful termination)."""
    gathered = gathered or []
    tab = evsi_table(p, exposure, costs)
    stop, why = stopping_reason(p, ci, exposure, costs, evidence)
    g, v = best_gather(p, exposure, costs)

    if stop:
        a, ec = choose_action(p, exposure, costs)
        return GateDecision(action=a, expected_cost=ec, gather=False,
                            evsi_best=v, evsi_table=tab, stop_reason=why,
                            ci=ci, p=p)
    if g is not None and rounds_done < max_rounds and g not in gathered:
        return GateDecision(action=g, expected_cost=0.0, gather=True,
                            evsi_best=v, evsi_table=tab, stop_reason="",
                            ci=ci, p=p)
    # Budget exhausted or already tried: act.
    a, ec = choose_action(p, exposure, costs)
    reason = why or f"budget_exhausted_after_{rounds_done}_rounds"
    return GateDecision(action=a, expected_cost=ec, gather=False,
                        evsi_best=v, evsi_table=tab, stop_reason=reason,
                        ci=ci, p=p)
