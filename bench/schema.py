"""JSON Schema for answer files, derived from spec section 6a / README 'Answer Format'.

Cross-field semantic rules the README states in prose
(`pattern_description` required iff pattern == 'undocumented'; SAR fields
blank when file == false; `sar.file` must agree with FILE_REPORT in final
actions; legitimate verdict implies empty affected_txn_ids / zero exposure)
are enforced in ``bench.validate.validate_answer`` because they are awkward
or impossible to express portably in JSON Schema.
"""

from __future__ import annotations

from typing import Any, Dict

PATTERNS = [
    "card_testing",
    "card_not_present_fraud",
    "card_not_present_new_device",
    "out_of_region_use",
    "account_takeover",
    "undocumented",
    "none",
]

ACTIONS = [
    "ALLOW_TRANSACTION",
    "DECLINE_TRANSACTION",
    "MONITOR_CARD",
    "MONITOR_CONNECTED_CARDS",
    "WARN_CUSTOMER",
    "VERIFY_WITH_CUSTOMER",
    "STEP_UP_AUTH",
    "BLOCK_CARD",
    "BLOCK_ALL_CARDS",
    "GENERATE_REPORT",
    "CREATE_CASE",
    "FILE_REPORT",
    "ESCALATE_TO_ANALYST",
    "CLOSE_NO_FRAUD",
]

_string_list = {"type": "array", "items": {"type": "string"}}

_evidence_item = {
    "type": "object",
    "required": ["claim", "source", "ref", "entity_ids"],
    "additionalProperties": False,
    "properties": {
        "claim": {"type": "string", "minLength": 1},
        "source": {"enum": ["graph", "document", "customer", "external"]},
        "ref": {"type": "string", "minLength": 1},
        "entity_ids": _string_list,
    },
}

_action_item = {
    "type": "object",
    "required": ["action", "route", "reason"],
    "additionalProperties": False,
    "properties": {
        "action": {"enum": ACTIONS},
        "route": {"enum": ["auto", "L1", "L2"]},
        "reason": {"type": "string", "minLength": 1},
    },
}

_evidence_request = {
    "type": "object",
    "required": ["type", "asked_after_step", "assumed_response"],
    "additionalProperties": False,
    "properties": {
        "type": {"enum": ["customer_validation", "step_up_auth", "analyst_info"]},
        "asked_after_step": {"type": "integer", "minimum": 0},
        "assumed_response": {"type": "string", "minLength": 1},
    },
}

_case = {
    "type": "object",
    "required": [
        "status",
        "verdict",
        "fraud_probability",
        "pattern",
        "pattern_description",
        "affected_txn_ids",
        "first_suspicious_txn_id",
        "connected_card_ids",
        "connected_device_profiles",
        "exposure_usd",
        "evidence",
        "similar_prior_cases",
        "summary",
        "written_to_graph",
        "graph_case_id",
    ],
    "additionalProperties": False,
    "properties": {
        "status": {"enum": ["open", "closed_fraud", "closed_legitimate", "escalated"]},
        "verdict": {"enum": ["fraud", "legitimate", "uncertain"]},
        "fraud_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "pattern": {"enum": PATTERNS},
        "pattern_description": {"type": "string"},
        "affected_txn_ids": _string_list,
        "first_suspicious_txn_id": {"type": "string"},
        "connected_card_ids": _string_list,
        "connected_device_profiles": _string_list,
        "exposure_usd": {"type": "number", "minimum": 0.0},
        "evidence": {"type": "array", "items": _evidence_item},
        "similar_prior_cases": _string_list,
        "summary": {"type": "string"},
        "written_to_graph": {"type": "boolean"},
        "graph_case_id": {"type": "string"},
    },
}

_sar = {
    "type": "object",
    "required": ["file", "reason", "narrative", "subjects", "total_amount_usd", "activity_dates"],
    "additionalProperties": False,
    "properties": {
        "file": {"type": "boolean"},
        "reason": {"type": "string", "minLength": 1},
        "narrative": {"type": "string"},
        "subjects": _string_list,
        "total_amount_usd": {"type": "number", "minimum": 0.0},
        "activity_dates": {
            "type": "array",
            "items": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
            "maxItems": 2,
        },
    },
}

_next_best_actions = {
    "type": "object",
    "required": ["initial", "final", "what_changed"],
    "additionalProperties": False,
    "properties": {
        "initial": {"type": "array", "items": _action_item, "minItems": 1},
        "final": {"type": "array", "items": _action_item, "minItems": 1},
        "what_changed": {"type": "string"},
    },
}

ANSWER_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Sumora HHGOA answer file",
    "type": "object",
    "required": [
        "case_id",
        "case",
        "evidence_requests",
        "next_best_actions",
        "sar",
        "stop_reason",
        "tool_calls",
        "tokens",
        "latency_s",
    ],
    "additionalProperties": False,
    "properties": {
        "case_id": {"type": "string", "pattern": "^HHG-\\d{3}$"},
        "case": _case,
        "evidence_requests": {"type": "array", "items": _evidence_request},
        "next_best_actions": _next_best_actions,
        "sar": _sar,
        "stop_reason": {"type": "string", "minLength": 1},
        "tool_calls": {"type": "integer", "minimum": 0},
        "tokens": {"type": "integer", "minimum": 0},
        "latency_s": {"type": "number", "minimum": 0.0},
    },
}


def get_schema() -> Dict[str, Any]:
    return ANSWER_SCHEMA
