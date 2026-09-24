"""Unit tests for the policy engine: one per rule per boundary value (spec section 6)."""

import pytest

from actions_api.models import ActionName, ActionRequest, Route
from actions_api import policy


def req(**kw):
    kw.setdefault("case_id", "HHG-001")
    kw.setdefault("txn_id", "3514030")
    return ActionRequest(**kw)


class TestRoutes:
    @pytest.mark.parametrize(
        "action",
        [a for a in ActionName if a in policy.AUTO_ACTIONS],
    )
    def test_auto_actions(self, action):
        assert policy.required_route(action) is Route.AUTO

    def test_decline_always_l1(self):
        assert policy.required_route(ActionName.DECLINE_TRANSACTION) is Route.L1

    @pytest.mark.parametrize(
        "exposure,expected",
        [
            (2499.99, Route.L1),
            (2500.00, Route.L1),  # boundary: 'exceeds' is strictly greater
            (2500.01, Route.L2),
        ],
    )
    def test_block_card_threshold(self, exposure, expected):
        assert policy.required_route(ActionName.BLOCK_CARD, exposure_usd=exposure) is expected

    def test_block_all_cards_always_l2(self):
        assert policy.required_route(ActionName.BLOCK_ALL_CARDS) is Route.L2

    def test_file_report_always_l2(self):
        assert policy.required_route(ActionName.FILE_REPORT, exposure_usd=0.0) is Route.L2
        assert policy.required_route(ActionName.FILE_REPORT, exposure_usd=99999.0) is Route.L2


class TestValidate:
    def test_route_mismatch_rejected(self):
        with pytest.raises(policy.PolicyError):
            policy.validate(
                ActionName.BLOCK_CARD,
                req(card_id="C12382-K1", exposure_usd=3000.0, proposed_route=Route.L1),
            )

    def test_route_match_accepted(self):
        route = policy.validate(
            ActionName.BLOCK_CARD,
            req(card_id="C12382-K1", exposure_usd=3000.0, proposed_route=Route.L2),
        )
        assert route is Route.L2

    def test_block_all_cards_r10_guard(self):
        with pytest.raises(policy.PolicyError) as ei:
            policy.validate(
                ActionName.BLOCK_ALL_CARDS,
                req(customer_id="C12382", card_ids=["C12382-K1", "C12382-K2"]),
            )
        assert ei.value.rule_id == "R10"

    def test_block_all_cards_r10_satisfied_by_two_fraud_cards(self):
        route = policy.validate(
            ActionName.BLOCK_ALL_CARDS,
            req(
                customer_id="C12382",
                card_ids=["C12382-K1", "C12382-K2"],
                customer_confirmed_fraud_cards=2,
            ),
        )
        assert route is Route.L2

    def test_block_all_cards_r10_satisfied_by_credentials(self):
        route = policy.validate(
            ActionName.BLOCK_ALL_CARDS,
            req(customer_id="C12382", card_ids=["C12382-K1"], credentials_compromised=True),
        )
        assert route is Route.L2

    def test_unknown_rule_id_rejected(self):
        with pytest.raises(policy.PolicyError):
            policy.validate(ActionName.MONITOR_CARD, req(card_id="C1", rule_ids=["R99"]))


class TestRuleGuards:
    def test_r1_boundary_at_0_70(self):
        assert "R1" in policy.fired_rules({"single_signal": True, "fraud_probability": 0.69})
        assert "R1" not in policy.fired_rules({"single_signal": True, "fraud_probability": 0.70})

    def test_r8_uncertain_boundary_at_500(self):
        assert "R8" not in policy.fired_rules({"verdict": "uncertain", "exposure_usd": 500.0})
        assert "R8" in policy.fired_rules({"verdict": "uncertain", "exposure_usd": 500.01})
        assert "R8" not in policy.fired_rules({"verdict": "fraud", "exposure_usd": 9999.0})
        assert "R8" in policy.fired_rules({"conflicting_evidence": True})

    def test_r10(self):
        assert "R10" in policy.fired_rules({"customer_confirmed_fraud_cards": 2})
        assert "R10" not in policy.fired_rules({"customer_confirmed_fraud_cards": 1})
        assert "R10" in policy.fired_rules({"credentials_compromised": True})

    def test_all_ten_rules_registered(self):
        assert set(policy.RULES) == {f"R{i}" for i in range(1, 11)}
