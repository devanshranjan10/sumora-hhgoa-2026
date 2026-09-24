"""Customer-persona responders (spec section 8).

The dataset does NOT provide customer or analyst replies (README section 5).
We simulate them:

- ``ScriptedPersonaResponder`` — deterministic mode for benchmark runs. Reads a
  fixtures JSON file keyed by case_id, e.g.::

      {"HHG-003": {"customer_validation": {"response": "I did not make this purchase...",
                                            "denies_transaction": true},
                   "step_up_auth": {"response": "...", "authenticated": false}}}

  Missing keys fall back to a seeded-random but fully deterministic response,
  so ``--deterministic`` benchmark runs are byte-identical with or without a
  complete fixtures file.

- ``LLMPersonaResponder`` — stub interface for the live demo; swap in a real
  LLMProvider later without touching callers.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

PERSONA_KINDS = ("customer_validation", "step_up_auth")


class PersonaResponder(Protocol):
    """Interface for simulating customer replies to evidence requests."""

    def respond(
        self,
        case_id: str,
        kind: str,
        question: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a response dict. Must include ``response`` (the assumed reply
        text recorded in the answer file's ``evidence_requests``)."""
        ...


class ScriptedPersonaResponder:
    """Deterministic responder backed by a fixtures file + seeded RNG fallback."""

    def __init__(self, fixtures_path: Optional[str] = None, seed: int = 42):
        self.seed = seed
        self.fixtures: Dict[str, Any] = {}
        if fixtures_path:
            p = Path(fixtures_path)
            if p.exists():
                self.fixtures = json.loads(p.read_text())

    def respond(
        self,
        case_id: str,
        kind: str,
        question: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        scripted = self.fixtures.get(case_id, {}).get(kind)
        if scripted:
            out = dict(scripted)
            out.setdefault("source", "scripted_fixture")
            return out
        return self._synthetic(case_id, kind)

    def _synthetic(self, case_id: str, kind: str) -> Dict[str, Any]:
        # Seeded on the case id so re-runs are byte-identical for any case set.
        rng = random.Random(f"{self.seed}:{case_id}:{kind}")
        if kind == "customer_validation":
            denies = rng.random() < 0.6
            response = (
                "Customer states they did not make this purchase and still has the card"
                if denies
                else "Customer confirms the purchase was theirs"
            )
            return {
                "source": "synthetic_seeded",
                "response": response,
                "denies_transaction": denies,
                "confirmed_transaction": not denies,
            }
        if kind == "step_up_auth":
            ok = rng.random() < 0.5
            return {
                "source": "synthetic_seeded",
                "response": (
                    "Customer completed the one-time passcode challenge"
                    if ok
                    else "Step-up authentication failed; no valid passcode presented"
                ),
                "authenticated": ok,
            }
        raise ValueError(f"unknown persona kind: {kind}")


class LLMPersonaResponder:
    """Stub interface for LLM-driven persona replies (live demo mode).

    A real implementation injects an LLMProvider (spec section 6b). The stub
    raises so nobody mistakes it for a working path; tests only check the
    interface shape.
    """

    def __init__(self, llm_provider: Any = None):
        self.llm_provider = llm_provider

    def respond(
        self,
        case_id: str,
        kind: str,
        question: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.llm_provider is None:
            raise NotImplementedError(
                "LLMPersonaResponder requires an LLMProvider; use "
                "ScriptedPersonaResponder for deterministic runs"
            )
        prompt = (
            f"You are the cardholder in fraud case {case_id}. An investigator asks: "
            f"{question!r}. Reply in one or two sentences, in character."
        )
        text = self.llm_provider.complete(prompt)  # type: ignore[attr-defined]
        return {"source": "llm", "response": text}
