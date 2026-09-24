"""Policy engine for the Sumora mock action APIs (README section 2/3, spec section 8).

Policy-as-code: the approval route for each of the 14 actions is decided here,
never by the caller. R1-R10 are encoded as named guard functions; the benchmark
runner uses RULES to validate that every rule_id cited in an answer file
exists, and ``fired_rules`` to validate that a cited rule's guard actually
holds for the case state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from .models import ActionName, ActionRequest, Route

# README section 2: BLOCK_CARD routes L2 when exposure exceeds $2,500.
BLOCK_CARD_L2_THRESHOLD_USD = 2500.0

# README section 2 approval routing table.
AUTO_ACTIONS: Set[ActionName] = {
    ActionName.ALLOW_TRANSACTION,
    ActionName.MONITOR_CARD,
    ActionName.MONITOR_CONNECTED_CARDS,
    ActionName.WARN_CUSTOMER,
    ActionName.VERIFY_WITH_CUSTOMER,
    ActionName.STEP_UP_AUTH,
    ActionName.GENERATE_REPORT,
    ActionName.CREATE_CASE,
    ActionName.ESCALATE_TO_ANALYST,
    ActionName.CLOSE_NO_FRAUD,
}
L1_ACTIONS: Set[ActionName] = {ActionName.DECLINE_TRANSACTION}
# BLOCK_ALL_CARDS and FILE_REPORT are ALWAYS L2; BLOCK_CARD depends on exposure.


class PolicyError(Exception):
    """Raised when a request violates the policy. Carries the violated rule id."""

    def __init__(self, message: str, rule_id: str = "POLICY"):
        super().__init__(message)
        self.rule_id = rule_id

# ---------------------------------------------------------------------------
# Rule registry R1-R10 (README section 3). Each guard receives the agent's
# case state and returns True when the rule's condition actually holds. The
# benchmark validator uses these to prove a cited rule fired.
# ---------------------------------------------------------------------------


def _r1(state: Dict[str, Any]) -> bool:
    """R1: single weak signal -> verify/step-up before any block.

    Fires when the assessed fraud probability is below 0.70 at EITHER decision
    point (initial recommendation or final), since R1 governs the verify-before-
    block discipline across the whole episode.
    """
    if not state.get("single_signal"):
        return False
    p_final = float(state.get("fraud_probability", 1.0))
    p_init = state.get("fraud_probability_initial")
    p_init = float(p_init) if p_init is not None else p_final
    return min(p_final, p_init) < 0.70


def _r2(state: Dict[str, Any]) -> bool:
    """R2: customer denied the transaction."""
    return bool(state.get("customer_denied"))


def _r3(state: Dict[str, Any]) -> bool:
    """R3: customer confirmed the transaction."""
    return bool(state.get("customer_confirmed"))


def _r4(state: Dict[str, Any]) -> bool:
    """R4: no customer reply within 24 hours."""
    return bool(state.get("no_reply_24h"))


def _r5(state: Dict[str, Any]) -> bool:
    """R5: card-testing sequence observed (>=3 small auths in 1h + larger purchase)."""
    return bool(state.get("card_testing_sequence"))


def _r6(state: Dict[str, Any]) -> bool:
    """R6: several cards share an origin (device profile / region / recipient email)."""
    return bool(state.get("shared_origin"))


def _r7(state: Dict[str, Any]) -> bool:
    """R7: disputed charge matches the customer's own recurring pattern."""
    return bool(state.get("matches_recurring_pattern"))


def _r8(state: Dict[str, Any]) -> bool:
    """R8: uncertain verdict with exposure > $500, or conflicting evidence."""
    exposure = float(state.get("exposure_usd", 0.0))
    uncertain_exposed = state.get("verdict") == "uncertain" and exposure > 500.0
    return bool(uncertain_exposed or state.get("conflicting_evidence"))


def _r9(state: Dict[str, Any]) -> bool:
    """R9: coordinated or repeated abuse fitting no documented pattern."""
    return state.get("pattern") == "undocumented" and bool(state.get("coordinated_abuse"))


def _r10(state: Dict[str, Any]) -> bool:
    """R10: BLOCK_ALL_CARDS only with >=2 confirmed-fraud cards or compromised credentials."""
    return int(state.get("customer_confirmed_fraud_cards", 0)) >= 2 or bool(
        state.get("credentials_compromised")
    )


RULES: Dict[str, Dict[str, Any]] = {
    "R1": {"description": "Verify before you block on a weak signal", "guard": _r1},
    "R2": {"description": "Customer denies the transaction", "guard": _r2},
    "R3": {"description": "Customer confirms the transaction", "guard": _r3},
    "R4": {"description": "No reply within 24 hours", "guard": _r4},
    "R5": {"description": "Card testing", "guard": _r5},
    "R6": {"description": "Shared origin", "guard": _r6},
    "R7": {"description": "Disputed but legitimate", "guard": _r7},
    "R8": {"description": "Escalate when uncertain and exposed", "guard": _r8},
    "R9": {"description": "Undocumented patterns", "guard": _r9},
    "R10": {"description": "Never BLOCK_ALL_CARDS without confirmed compromise", "guard": _r10},
}


def rule_exists(rule_id: str) -> bool:
    return rule_id in RULES


def fired_rules(state: Dict[str, Any]) -> List[str]:
    """Return the ids of every rule whose guard holds for the given case state."""
    return [rid for rid, spec in RULES.items() if spec["guard"](state)]


# ---------------------------------------------------------------------------
# Route decision
# ---------------------------------------------------------------------------


def required_route(action: ActionName, exposure_usd: float = 0.0) -> Route:
    """Authoritative approval route per README section 2."""
    if action in AUTO_ACTIONS:
        return Route.AUTO
    if action is ActionName.DECLINE_TRANSACTION:
        return Route.L1
    if action is ActionName.BLOCK_CARD:
        # Strictly greater than $2,500 routes L2 (boundary $2,500.00 is L1).
        return Route.L2 if exposure_usd > BLOCK_CARD_L2_THRESHOLD_USD else Route.L1
    if action in (ActionName.BLOCK_ALL_CARDS, ActionName.FILE_REPORT):
        return Route.L2
    raise PolicyError(f"no route defined for action {action}")  # pragma: no cover


def validate(action: ActionName, request: ActionRequest) -> Route:
    """Validate a proposed action against the policy and return the enforced route.

    Raises PolicyError when:
      - the proposed route disagrees with the authoritative route;
      - BLOCK_ALL_CARDS is requested without satisfying R10;
      - an unknown rule_id is cited.
    """
    route = required_route(action, request.exposure_usd)

    if action is ActionName.BLOCK_ALL_CARDS and not _r10(
        {
            "customer_confirmed_fraud_cards": request.customer_confirmed_fraud_cards,
            "credentials_compromised": request.credentials_compromised,
        }
    ):
        raise PolicyError(
            "R10: BLOCK_ALL_CARDS requires >=2 confirmed-fraud cards or confirmed "
            "credential compromise",
            rule_id="R10",
        )

    if request.proposed_route is not None and request.proposed_route is not route:
        raise PolicyError(
            f"proposed route {request.proposed_route.value} does not match required "
            f"route {route.value} for {action.value}"
            + (
                f" (exposure ${request.exposure_usd:.2f} vs "
                f"${BLOCK_CARD_L2_THRESHOLD_USD:.2f} threshold)"
                if action is ActionName.BLOCK_CARD
                else ""
            )
        )

    for rid in request.rule_ids:
        if not rule_exists(rid):
            raise PolicyError(f"unknown rule_id cited: {rid}")

    return route
