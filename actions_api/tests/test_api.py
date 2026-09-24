"""End-to-end API tests: all 14 endpoints, policy enforcement, decision store, personas."""

import json

import pytest
from fastapi.testclient import TestClient

from actions_api.main import ENDPOINTS, create_app
from actions_api.store import InMemoryDecisionStore, JsonlDecisionStore
from actions_api.personas import ScriptedPersonaResponder, LLMPersonaResponder


@pytest.fixture()
def store():
    return InMemoryDecisionStore()


@pytest.fixture()
def client(store):
    return TestClient(create_app(store=store))


def base_payload(**kw):
    p = {"case_id": "HHG-002", "txn_id": "3478782", "card_id": "C11891-K1",
         "customer_id": "C11891", "reason": "test"}
    p.update(kw)
    return p


class TestAllEndpoints:
    def test_all_14_endpoints_registered(self, client):
        paths = {r.path for r in client.app.routes}
        for path in ENDPOINTS:
            assert f"/actions/{path}" in paths
        assert len(ENDPOINTS) == 14

    @pytest.mark.parametrize("path", list(ENDPOINTS))
    def test_each_endpoint_happy_path(self, client, path):
        extra = {}
        if path == "block_all_cards":
            extra = {"card_ids": ["C11891-K1"], "credentials_compromised": True}
        if path == "monitor_connected_cards":
            extra = {"card_ids": ["C11891-K2"]}
        resp = client.post(f"/actions/{path}", json=base_payload(**extra))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is True
        assert body["route"] in ("auto", "L1", "L2")
        assert body["executed"] == (body["route"] == "auto")
        assert body["simulated_latency_ms"] > 0
        assert body["decision_id"].startswith("DEC-")

    def test_block_card_route_enforced_by_exposure(self, client):
        r = client.post("/actions/block_card", json=base_payload(exposure_usd=100.0))
        assert r.json()["route"] == "L1"
        r = client.post("/actions/block_card", json=base_payload(exposure_usd=9000.0))
        assert r.json()["route"] == "L2"

    def test_policy_violation_returns_422(self, client):
        r = client.post(
            "/actions/block_card",
            json=base_payload(exposure_usd=9000.0, proposed_route="L1"),
        )
        assert r.status_code == 422
        assert "required route L2" in r.json()["detail"]

    def test_block_all_cards_r10_returns_422(self, client):
        r = client.post("/actions/block_all_cards", json=base_payload())
        assert r.status_code == 422
        assert r.json()["rule_id"] == "R10"

    def test_decision_written_to_store(self, client, store):
        client.post("/actions/monitor_card", json=base_payload())
        client.post("/actions/file_report", json=base_payload())
        decisions = store.list_for_case("HHG-002")
        assert [d.action.value for d in decisions] == ["MONITOR_CARD", "FILE_REPORT"]
        assert decisions[0].status == "executed"  # auto route
        assert decisions[1].status == "pending_approval"  # L2 waits for human

    def test_decisions_endpoint(self, client):
        client.post("/actions/create_case", json=base_payload())
        r = client.get("/decisions", params={"case_id": "HHG-002"})
        assert r.json()["count"] == 1


class TestPersonas:
    def test_scripted_fixture_used(self, tmp_path):
        fixtures = tmp_path / "personas.json"
        fixtures.write_text(json.dumps({
            "HHG-002": {"customer_validation": {
                "response": "I did not make this purchase", "denies_transaction": True}}
        }))
        store = InMemoryDecisionStore()
        client = TestClient(create_app(
            store=store, persona=ScriptedPersonaResponder(str(fixtures))))
        r = client.post("/actions/verify_with_customer", json=base_payload())
        reply = r.json()["result"]["persona_reply"]
        assert reply["response"] == "I did not make this purchase"
        assert reply["denies_transaction"] is True

    def test_scripted_fallback_is_deterministic(self):
        p1 = ScriptedPersonaResponder(seed=7)
        p2 = ScriptedPersonaResponder(seed=7)
        a = p1.respond("HHG-009", "customer_validation", "did you?")
        b = p2.respond("HHG-009", "customer_validation", "did you?")
        assert a == b
        assert "response" in a

    def test_step_up_auth_persona(self, client):
        r = client.post("/actions/step_up_auth", json=base_payload())
        reply = r.json()["result"]["persona_reply"]
        assert "authenticated" in reply

    def test_llm_persona_stub_interface(self):
        persona = LLMPersonaResponder()  # no provider wired
        with pytest.raises(NotImplementedError):
            persona.respond("HHG-001", "customer_validation", "did you?")

        class FakeLLM:
            def complete(self, prompt):
                return "Yes, that was me at the grocery store."

        out = LLMPersonaResponder(FakeLLM()).respond("HHG-001", "customer_validation", "did you?")
        assert out["source"] == "llm"
        assert "grocery" in out["response"]


class TestJsonlStore:
    def test_roundtrip(self, tmp_path):
        store = InMemoryDecisionStore()
        client = TestClient(create_app(store=store))
        client.post("/actions/create_case", json=base_payload())
        jsonl = JsonlDecisionStore(str(tmp_path / "decisions.jsonl"))
        for d in store.all():
            jsonl.write(d)
        assert jsonl.list_for_case("HHG-002")[0].action.value == "CREATE_CASE"
