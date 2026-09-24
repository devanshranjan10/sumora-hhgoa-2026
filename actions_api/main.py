"""FastAPI app for the 14 mock policy actions (spec section 8).

Each endpoint: validates the request against the policy engine (approval-route
check) -> writes a Decision record to the pluggable store -> returns a
realistic JSON response with simulated latency. Customer-validation and
step-up-auth replies are simulated by a persona responder (README rule:
responses are not provided by the dataset; the assumption is recorded).
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import policy
from .models import ActionName, ActionRequest, ActionResponse, Decision, Route
from .personas import PersonaResponder, ScriptedPersonaResponder
from .store import DecisionStore, InMemoryDecisionStore

# Endpoint path -> action enum.
ENDPOINTS: Dict[str, ActionName] = {
    "allow_transaction": ActionName.ALLOW_TRANSACTION,
    "decline_transaction": ActionName.DECLINE_TRANSACTION,
    "monitor_card": ActionName.MONITOR_CARD,
    "monitor_connected_cards": ActionName.MONITOR_CONNECTED_CARDS,
    "warn_customer": ActionName.WARN_CUSTOMER,
    "verify_with_customer": ActionName.VERIFY_WITH_CUSTOMER,
    "step_up_auth": ActionName.STEP_UP_AUTH,
    "block_card": ActionName.BLOCK_CARD,
    "block_all_cards": ActionName.BLOCK_ALL_CARDS,
    "generate_report": ActionName.GENERATE_REPORT,
    "create_case": ActionName.CREATE_CASE,
    "file_report": ActionName.FILE_REPORT,
    "escalate_to_analyst": ActionName.ESCALATE_TO_ANALYST,
    "close_no_fraud": ActionName.CLOSE_NO_FRAUD,
}

PERSONA_KIND_BY_ACTION = {
    ActionName.VERIFY_WITH_CUSTOMER: "customer_validation",
    ActionName.STEP_UP_AUTH: "step_up_auth",
}


def create_app(
    store: Optional[DecisionStore] = None,
    persona: Optional[PersonaResponder] = None,
    latency_seed: int = 20260919,
) -> FastAPI:
    """App factory so tests can inject their own store/persona."""
    app = FastAPI(title="Sumora Mock Action APIs", version="0.1.0")
    app.state.store = store or InMemoryDecisionStore()
    app.state.persona = persona or ScriptedPersonaResponder(seed=latency_seed)
    app.state.latency_rng = random.Random(latency_seed)

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.exception_handler(policy.PolicyError)
    def policy_error_handler(_req: Request, exc: policy.PolicyError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": str(exc),
                "rule_id": exc.rule_id,
                "action": "",
                "proposed_route": None,
                "required_route": None,
            },
        )

    def make_handler(action: ActionName):
        def handler(body: ActionRequest) -> ActionResponse:
            t0 = time.perf_counter()
            route = policy.validate(action, body)
            result = _simulate_outcome(action, body, app.state.persona)
            latency_ms = (time.perf_counter() - t0) * 1000.0 + _simulated_service_latency(
                action, app.state.latency_rng
            )
            executed = route is Route.AUTO
            decision = Decision(
                case_id=body.case_id,
                action=action,
                route=route,
                rule_ids=body.rule_ids,
                reason=body.reason,
                status="executed" if executed else "pending_approval",
                outcome=result,
                latency_ms=round(latency_ms, 2),
            )
            app.state.store.write(decision)
            return ActionResponse(
                ok=True,
                action=action,
                route=route,
                executed=executed,
                decision_id=decision.decision_id,
                simulated_latency_ms=decision.latency_ms,
                result=result,
            )

        return handler

    for path, action in ENDPOINTS.items():
        app.post(
            f"/actions/{path}", response_model=ActionResponse, name=action.value
        )(make_handler(action))

    @app.get("/decisions")
    def list_decisions(case_id: Optional[str] = None):
        st: DecisionStore = app.state.store
        decisions = st.list_for_case(case_id) if case_id else st.all()
        return {
            "count": len(decisions),
            "decisions": [d.model_dump(mode="json") for d in decisions],
        }

    return app


def _simulated_service_latency(action: ActionName, rng: random.Random) -> float:
    """Realistic per-action latency in ms (deterministic for a fixed seed)."""
    heavy = {ActionName.GENERATE_REPORT, ActionName.FILE_REPORT, ActionName.CREATE_CASE}
    lo, hi = (180.0, 650.0) if action in heavy else (25.0, 160.0)
    return round(rng.uniform(lo, hi), 2)


def _simulate_outcome(
    action: ActionName, body: ActionRequest, persona: PersonaResponder
) -> Dict[str, Any]:
    """Realistic per-action result payloads. All integrations are mocks (spec non-goal)."""
    if action in PERSONA_KIND_BY_ACTION:
        kind = PERSONA_KIND_BY_ACTION[action]
        reply = persona.respond(
            case_id=body.case_id,
            kind=kind,
            question=body.question or "Did you make this transaction?",
            context=body.model_dump(mode="json"),
        )
        return {"channel": "sms+app", "delivered": True, "persona_reply": reply}
    if action is ActionName.ALLOW_TRANSACTION:
        return {"txn_id": body.txn_id, "authorization": "allowed_to_stand"}
    if action is ActionName.DECLINE_TRANSACTION:
        return {"txn_id": body.txn_id, "authorization": "declined", "card_stays_active": True}
    if action is ActionName.MONITOR_CARD:
        return {"card_id": body.card_id, "monitoring_window_hours": 72, "sensitivity": "high"}
    if action is ActionName.MONITOR_CONNECTED_CARDS:
        return {"card_ids": body.card_ids, "monitoring_window_hours": 72, "sensitivity": "high"}
    if action is ActionName.WARN_CUSTOMER:
        return {
            "customer_id": body.customer_id,
            "message_sent": body.message or "",
            "channel": "app_push",
        }
    if action is ActionName.BLOCK_CARD:
        return {"card_id": body.card_id, "blocked": True, "reissue_ordered": True}
    if action is ActionName.BLOCK_ALL_CARDS:
        return {"customer_id": body.customer_id, "cards_blocked": body.card_ids}
    if action is ActionName.GENERATE_REPORT:
        return {"case_id": body.case_id, "report_type": "internal_investigation", "pages": 2}
    if action is ActionName.CREATE_CASE:
        return {"case_id": body.case_id, "opened": True, "write_target": "tigergraph"}
    if action is ActionName.FILE_REPORT:
        return {
            "case_id": body.case_id,
            "filing": "suspicious_activity_report",
            "regulator": "FinCEN",
        }
    if action is ActionName.ESCALATE_TO_ANALYST:
        return {"case_id": body.case_id, "queue": "fraud_analysts_l2", "sla_hours": 4}
    if action is ActionName.CLOSE_NO_FRAUD:
        return {"case_id": body.case_id, "closed_as": "legitimate"}
    raise policy.PolicyError(f"unhandled action {action}")  # pragma: no cover


# Module-level app for `uvicorn actions_api.main:app`.
app = create_app()
