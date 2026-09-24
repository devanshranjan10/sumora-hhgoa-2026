# Runtime architecture

Sumora runs the investigation API, the decision model, the official TigerGraph
MCP server, and TigerGraph Community Edition on one CPU VM. Nginx serves the
React dashboard and forwards `/api/` to the Python API. The browser never
connects to TigerGraph or MCP directly.

| Component | Owns | Reads from |
|---|---|---|
| `service/app.py` | HTTP cases, imports, health, and run storage | `agent/`, the raw CSV index, TigerGraph |
| `agent/runner.py` | LangGraph investigation state and stopping decisions | `agent/tools.py`, the decision model, the policy |
| `agent/decision_model.py` | Fraud probability and calibration interval | `eval/overlap_model/candidate.joblib`, raw transaction fields |
| `agent/mcp_graph.py` | Installed pattern and similarity queries | [TigerGraph MCP](https://github.com/tigergraph/tigergraph-mcp) on loopback |
| `agent/tools.py` | Bounded graph reads and audited graph writes | MCP for selected reads; RESTPP for other reads and writes |
| `actions_api/policy.py` | Action guards R1–R10 and approval routes | Case state |
| `ui/` | Graph, evidence, agent steps, imports, and watchlist | API and committed recorded traces |

The live agent opens a case, gathers graph and policy evidence, predicts `p`,
and compares the expected cost of acting with the value of gathering more
information. It then proposes policy actions, records the evidence and steps,
and writes the case back to TigerGraph. The answer writer validates the JSON
contract before accepting the output. `cases/` contains exactly the 20 scored
answers. Demo imports use separate run storage and cannot replace those files.

## Data boundaries

The raw challenge CSVs remain under `data/HHGOA_IEEE/` and outside Git.
`scripts/prep_load_data.py` builds derived columns for TigerGraph in
`data/HHGOA_IEEE/prepped/`. The decision model reads the raw files because
prepared numeric blanks differ from the training data. The API rejects a
prepared transaction file at startup.

The case pack and closed-case history supply graph links and historical
outcomes. Only closed cases with known outcomes enter model fitting. September
closed cases select the calibrator; October closed cases evaluate the selected
model. The 20 challenge outcomes are unknown. See
[model and graph evidence](evaluation.md) for counts and limits.

New transaction uploads write the transaction and its card, customer, and
device links into TigerGraph before the in-memory index accepts the row. A
new row on an existing card must follow that card's last transaction. The API
adds the chronological `NEXT` edge and persists the import definition for
restart replay. New-row estimates use the six-feature fallback and are marked
provisional because the raw source fields used by the rich model are absent.

## Deployment boundary

The API and MCP server listen on `127.0.0.1`; Nginx is the public entry point.
The MCP service exposes only `tigergraph__run_installed_query`. The deployed
model is CPU gradient boosting. The Qwen3-14B fine-tune is an audited
experiment and does not supply live verdicts or actions.

The hosted VM is a public demo, with rate limits and bounded imports. It does
not authenticate analysts or execute card blocks and regulatory filings.
Those actions remain recommendations with the approval route required by the
bank policy. See [the deployment guide](deployment.md) for boot, health, cost,
and graph load commands.
