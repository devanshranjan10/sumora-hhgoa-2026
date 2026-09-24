"""Sumora agent package (spec §6).

The real investigation loop. `runner.GraphAgentRunner` implements the
`bench.agent.AgentRunner` protocol, so the benchmark can run either the stub
(deterministic fallback) or this loop interchangeably.

Structure:
  calibrator.py  — isotonic-calibrated p (the ONLY uncertainty quantity)
  gate.py        — expected-cost action choice + EVSI gather-vs-act + stopping
  tools.py       — InvestigationTools facade over the graph (~14 primitives)
  runner.py      — GraphAgentRunner: the loop wired to the bench protocol
  llm.py         — LLMProvider protocol + deterministic offline fallback
"""
