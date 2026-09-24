"""LLM provider seam (spec §6c).

The loop talks to an LLM through this protocol only. Three implementations:
  VertexGeminiProvider - Vertex AI Gemini (VM service account, no key file)
  OllamaProvider       - local Ollama (the M1 production target)
  EchoProvider         - deterministic offline fallback for tests/bench

The LLM narrates and fills summary/SAR slots. It never emits probabilities
(the calibrator owns p) and never chooses actions (the gate owns a*).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Protocol


class LLMProvider(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 900) -> str: ...


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of an LLM reply (handles ```json fences)."""
    m = re.search(r"\{", text)
    if not m:
        return None
    depth = 0
    start = m.start()
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


class EchoProvider:
    """Deterministic fallback: returns a compact JSON echo of the request.
    Keeps bench runs and CI fully offline and reproducible."""

    def complete(self, system: str, user: str, max_tokens: int = 900) -> str:
        return json.dumps({"echo": user[:200], "system_hash": hash(system) % 10_000})


class VertexGeminiProvider:
    """Vertex AI via the VM's service-account identity (metadata server)."""

    def __init__(self, project: str = "contral-6b0bd", location: str = "us-central1",
                 model: str = "gemini-2.5-flash") -> None:
        import httpx  # local import: optional dependency
        self._httpx = httpx
        self.project = project
        self.location = location
        self.model = model
        self._token: Optional[str] = None

    def _auth(self, force: bool = False) -> str:
        if force or self._token is None:
            r = self._httpx.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/"
                "service-accounts/default/token",
                headers={"Metadata-Flavor": "Google"}, timeout=10,
            )
            r.raise_for_status()
            self._token = r.json()["access_token"]
        return self._token

    def complete(self, system: str, user: str, max_tokens: int = 900) -> str:
        url = (f"https://{self.location}-aiplatform.googleapis.com/v1/projects/"
               f"{self.project}/locations/{self.location}/publishers/google/"
               f"models/{self.model}:generateContent")
        body = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens},
        }
        r = self._httpx.post(url, json=body,
                             headers={"Authorization": f"Bearer {self._auth()}"},
                             timeout=120)
        if r.status_code == 401:
            # token expired: refresh once and retry
            self._token = None
            r = self._httpx.post(url, json=body,
                                 headers={"Authorization": f"Bearer {self._auth(force=True)}"},
                                 timeout=120)
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)


class OllamaProvider:
    """Local Ollama server (the 16GB M1 production target per spec §6b)."""

    def __init__(self, model: str = "qwen3:14b", base: str = "http://localhost:11434") -> None:
        self.model = model
        self.base = base

    def complete(self, system: str, user: str, max_tokens: int = 900) -> str:
        import httpx
        r = httpx.post(f"{self.base}/api/generate", json={
            "model": self.model, "prompt": user, "system": system, "stream": False,
            "options": {"temperature": 0.2, "num_predict": max_tokens},
        }, timeout=300)
        r.raise_for_status()
        return r.json()["response"]
