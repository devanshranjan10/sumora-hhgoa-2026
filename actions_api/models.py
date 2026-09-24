"""Pydantic models for the Sumora mock action APIs (spec section 8).

Every action request is validated against the policy engine, recorded as a
Decision record via a pluggable store, and answered with a realistic payload
plus simulated latency metadata.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ActionName(str, Enum):
    """The 14 policy actions (README section 3, spec section 8)."""

    ALLOW_TRANSACTION = "ALLOW_TRANSACTION"
    DECLINE_TRANSACTION = "DECLINE_TRANSACTION"
    MONITOR_CARD = "MONITOR_CARD"
    MONITOR_CONNECTED_CARDS = "MONITOR_CONNECTED_CARDS"
    WARN_CUSTOMER = "WARN_CUSTOMER"
    VERIFY_WITH_CUSTOMER = "VERIFY_WITH_CUSTOMER"
    STEP_UP_AUTH = "STEP_UP_AUTH"
    BLOCK_CARD = "BLOCK_CARD"
    BLOCK_ALL_CARDS = "BLOCK_ALL_CARDS"
    GENERATE_REPORT = "GENERATE_REPORT"
    CREATE_CASE = "CREATE_CASE"
    FILE_REPORT = "FILE_REPORT"
    ESCALATE_TO_ANALYST = "ESCALATE_TO_ANALYST"
    CLOSE_NO_FRAUD = "CLOSE_NO_FRAUD"


class Route(str, Enum):
    """Approval routes (README section 2)."""

    AUTO = "auto"
    L1 = "L1"
    L2 = "L2"


class ActionRequest(BaseModel):
    """Request body for every action endpoint.

    `proposed_route` is what the agent believes the approval route is; the
    policy engine recomputes the authoritative route and rejects a mismatch.
    """

    case_id: str = Field(..., description="Case this action belongs to, e.g. HHG-001")
    proposed_route: Optional[Route] = Field(
        default=None,
        description="Route the agent proposes; validated against the policy engine",
    )
    rule_ids: List[str] = Field(
        default_factory=list,
        description="Policy rules the agent cites, e.g. ['R1', 'R5']",
    )
    txn_id: Optional[str] = Field(default=None, description="Flagged transaction id")
    card_id: Optional[str] = Field(default=None, description="Card the action targets")
    customer_id: Optional[str] = Field(default=None, description="Customer id")
    card_ids: List[str] = Field(
        default_factory=list, description="Card set for BLOCK_ALL_CARDS / MONITOR_CONNECTED_CARDS"
    )
    exposure_usd: float = Field(
        default=0.0, ge=0.0, description="Case exposure; drives the BLOCK_CARD $2,500 threshold"
    )
    fraud_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    customer_confirmed_fraud_cards: int = Field(
        default=0, ge=0, description="Cards with confirmed fraud (R10 guard for BLOCK_ALL_CARDS)"
    )
    credentials_compromised: bool = Field(
        default=False, description="Customer credentials confirmed compromised (R10 guard)"
    )
    question: Optional[str] = Field(
        default=None, description="Question posed for VERIFY_WITH_CUSTOMER / STEP_UP_AUTH"
    )
    message: Optional[str] = Field(default=None, description="Message text for WARN_CUSTOMER")
    reason: str = Field(default="", description="Free-text reason citing policy rules")

    @model_validator(mode="after")
    def _check_target_ids(self) -> "ActionRequest":
        # Endpoint-specific requirements are enforced in main.py; here we only
        # sanity-check that at least one target identifier is present.
        if not (self.txn_id or self.card_id or self.customer_id or self.card_ids):
            raise ValueError("at least one of txn_id, card_id, customer_id, card_ids is required")
        return self


class Decision(BaseModel):
    """The decision record written to the store for every action (spec section 8)."""

    decision_id: str = Field(default_factory=lambda: f"DEC-{uuid.uuid4().hex[:12]}")
    case_id: str
    action: ActionName
    route: Route
    rule_ids: List[str]
    reason: str
    status: Literal["executed", "pending_approval"]
    outcome: Dict[str, Any]
    decided_at_epoch: float = Field(default_factory=time.time)
    latency_ms: float


class ActionResponse(BaseModel):
    """Uniform response envelope for all 14 endpoints."""

    ok: bool
    action: ActionName
    route: Route
    executed: bool = Field(
        ..., description="True only for auto-route actions; L1/L2 are recommended and wait"
    )
    decision_id: str
    simulated_latency_ms: float
    result: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class PolicyViolation(BaseModel):
    detail: str
    action: str
    proposed_route: Optional[str] = None
    required_route: Optional[str] = None
