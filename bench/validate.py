"""Answer-file validation (spec section 6a: validate before accepting).

Layers:
1. JSON Schema (bench.schema.ANSWER_SCHEMA).
2. Semantic cross-field rules the README states in prose.
3. ID existence: every cited ID must exist in the dataset (README: 'Made-up
   IDs score zero').
4. rule_id-fired: every rule cited in an action reason must exist in the
   policy registry and, when ``case_states`` are provided, its guard must
   actually hold for that case.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import jsonschema

from actions_api import policy
from bench.dataset import (
    CASE_ID_RE,
    CLOSED_CASE_RE,
    CARD_RE,
    CUSTOMER_RE,
    DatasetIndex,
    TXN_RE,
)

RULE_REF_RE = re.compile(r"\bR(?:10|[1-9])\b")
# Broader scan for rule-like tokens (R42, R999, ...) so malformed or unknown
# rule citations are caught, not silently skipped by the strict pattern.
RULE_TOKEN_RE = re.compile(r"\bR\d+\b")

SAR_TRIGGER_ACTIONS = {"FILE_REPORT"}


def validate_schema(answer: Dict[str, Any]) -> List[str]:
    """Return JSON Schema violations (empty list = valid)."""
    from bench.schema import ANSWER_SCHEMA

    validator = jsonschema.Draft202012Validator(ANSWER_SCHEMA)
    return sorted(e.message for e in validator.iter_errors(answer))


def _cited_rules(nba: Dict[str, Any], sar_reason: str) -> List[str]:
    rules: List[str] = []
    for entry in nba["initial"] + nba["final"]:
        rules.extend(RULE_REF_RE.findall(entry["reason"]))
    rules.extend(RULE_REF_RE.findall(sar_reason))
    return sorted(set(rules))


def _all_rule_tokens(nba: Dict[str, Any], sar_reason: str) -> List[str]:
    tokens: List[str] = []
    for entry in nba["initial"] + nba["final"]:
        tokens.extend(RULE_TOKEN_RE.findall(entry["reason"]))
    tokens.extend(RULE_TOKEN_RE.findall(sar_reason))
    return sorted(set(tokens))


def validate_semantics(answer: Dict[str, Any]) -> List[str]:
    """Cross-field rules from the README 'Notes' and field tables."""
    errors: List[str] = []
    case = answer["case"]
    sar = answer["sar"]
    nba = answer["next_best_actions"]

    if case["pattern"] == "undocumented" and not case["pattern_description"].strip():
        errors.append("pattern_description is required when pattern is 'undocumented'")
    if case["pattern"] != "undocumented" and case["pattern_description"].strip():
        errors.append("pattern_description must be '' unless pattern is 'undocumented'")

    final_actions = {a["action"] for a in nba["final"]}
    files_report = "FILE_REPORT" in final_actions
    if sar["file"] != files_report:
        errors.append(
            f"sar.file ({sar['file']}) disagrees with FILE_REPORT in final actions ({files_report})"
        )

    if sar["file"]:
        if not sar["narrative"].strip():
            errors.append("sar.narrative is required when sar.file is true")
        if len(sar["activity_dates"]) != 2:
            errors.append("sar.activity_dates must have exactly two dates when filing")
        if not sar["subjects"]:
            errors.append("sar.subjects must not be empty when filing")
        if sar["total_amount_usd"] <= 0:
            errors.append("sar.total_amount_usd must be positive when filing")
    else:
        if sar["narrative"] != "" or sar["subjects"] != [] or sar["total_amount_usd"] != 0 or sar["activity_dates"] != []:
            errors.append("sar fields must be blank when sar.file is false")

    if case["verdict"] == "legitimate":
        if case["affected_txn_ids"]:
            errors.append("affected_txn_ids must be empty for a legitimate verdict")
        if case["exposure_usd"] != 0:
            errors.append("exposure_usd must be 0 for a legitimate verdict")
        if sar["file"]:
            errors.append("sar.file must be false for a legitimate verdict")

    if case["first_suspicious_txn_id"] and case["affected_txn_ids"]:
        if case["first_suspicious_txn_id"] not in case["affected_txn_ids"]:
            errors.append("first_suspicious_txn_id must be one of affected_txn_ids")

    if case["verdict"] == "fraud" and not case["affected_txn_ids"]:
        errors.append("affected_txn_ids must include at least the flagged txn for a fraud verdict")

    if not answer["evidence_requests"]:
        if nba["initial"] != nba["final"]:
            errors.append("final must equal initial when no evidence was requested")
        elif nba["what_changed"] not in ("nothing", ""):
            errors.append("what_changed must be 'nothing' when no evidence was requested")

    return errors


def validate_ids(answer: Dict[str, Any], idx: DatasetIndex) -> List[str]:
    """Every cited ID must exist in the dataset (README: made-up IDs score zero)."""
    errors: List[str] = []
    case = answer["case"]

    def check(value: str, where: str) -> None:
        if TXN_RE.match(value):
            if not idx.txn_exists(value):
                errors.append(f"{where}: transaction id {value} not in dataset")
        elif CLOSED_CASE_RE.match(value):
            if not idx.closed_case_exists(value):
                errors.append(f"{where}: closed case id {value} not in dataset")
        elif CARD_RE.match(value):
            if not idx.card_exists(value):
                errors.append(f"{where}: card id {value} not in dataset")
        elif CUSTOMER_RE.match(value):
            if not idx.customer_exists(value):
                errors.append(f"{where}: customer id {value} not in dataset")
        # anything else (e.g. CASE-2016-1187 graph ids, device strings) is not
        # a dataset entity reference and is not checked here.

    if not CASE_ID_RE.match(answer["case_id"]):
        errors.append(f"case_id {answer['case_id']} does not look like HHG-NNN")

    for t in case["affected_txn_ids"]:
        check(t, "case.affected_txn_ids")
    if case["first_suspicious_txn_id"]:
        check(case["first_suspicious_txn_id"], "case.first_suspicious_txn_id")
    for c in case["connected_card_ids"]:
        check(c, "case.connected_card_ids")
    for cc in case["similar_prior_cases"]:
        check(cc, "case.similar_prior_cases")
    for i, ev in enumerate(case["evidence"]):
        for eid in ev["entity_ids"]:
            check(eid, f"case.evidence[{i}].entity_ids")
    for s in answer["sar"]["subjects"]:
        check(s, "sar.subjects")
    for p in case["connected_device_profiles"]:
        if not idx.device_profile_exists(p):
            errors.append(
                f"case.connected_device_profiles: device profile {p!r} not in identity.csv"
            )
    return errors


def validate_rules(
    answer: Dict[str, Any], case_state: Optional[Dict[str, Any]] = None
) -> List[str]:
    """rule_id-fired validation: every cited rule must exist, and (when the
    case state is available) its guard must actually hold."""
    errors: List[str] = []
    nba = answer["next_best_actions"]
    # Unknown / malformed rule tokens first (e.g. R42, R0, R123).
    for token in _all_rule_tokens(nba, answer["sar"]["reason"]):
        if not policy.rule_exists(token):
            errors.append(f"cited rule {token} does not exist in the policy registry")
    cited = _cited_rules(nba, answer["sar"]["reason"])
    for rid in cited:
        if not policy.rule_exists(rid):
            errors.append(f"cited rule {rid} does not exist in the policy registry")
    if case_state is not None:
        fired = set(policy.fired_rules(case_state))
        for rid in cited:
            if rid in fired:
                continue
            # R10 is special: it constrains BLOCK_ALL_CARDS; citing it in a
            # reason is meaningful even when its guard is the thing being
            # evaluated. Treat it as 'fired' whenever BLOCK_ALL_CARDS appears.
            if rid == "R10" and any(
                a["action"] == "BLOCK_ALL_CARDS"
                for a in answer["next_best_actions"]["initial"]
                + answer["next_best_actions"]["final"]
            ):
                continue
            errors.append(f"cited rule {rid} did not fire for this case state")
    return errors


def validate_answer(
    answer: Dict[str, Any],
    idx: Optional[DatasetIndex] = None,
    case_state: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Full validation pipeline. Returns a list of errors ([] = accepted)."""
    errors = validate_schema(answer)
    if errors:
        return errors  # schema failure makes deeper checks unreliable
    errors.extend(validate_semantics(answer))
    if idx is not None:
        errors.extend(validate_ids(answer, idx))
    errors.extend(validate_rules(answer, case_state))
    return errors
