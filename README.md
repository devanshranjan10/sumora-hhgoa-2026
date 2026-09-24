# Sumora: agentic Fraud Investigation on TigerGraph

Hacker House Goa 2026 submission for the TigerGraph **Agentic Fraud Investigation**
partner task: an AI agent that investigates payment-fraud cases on a 590k-transaction
graph, quantifies its own uncertainty, decides when to gather more evidence vs act,
recommends next-best actions under a bank policy, and writes cases + memory back into
the graph.

## Architecture

```
case trigger ──► open_case ──► investigate ──► assess_uncertainty ──► EVSI gate
                     ▲                                │                  │
                     │              gather_evidence ◄─┘      propose ◄───┘
                     │                                              │
                memory ◄── execute ◄── approval ◄── explain ◄────────┘
```

- **`agent/`** is the LangGraph investigation loop. One uncertainty quantity: a
  gradient-boosting score over as-of account history and supplied transaction
  fields. September closed cases select a sigmoid calibrator for the rich model
  and isotonic calibration for the fallback; October remains held out.
  A six-feature model handles new rows without those fields. A deterministic
  explanation layer describes `p` without generating it. A gather-vs-act gate
  compares expected cost of acting now against expected value of sample
  information (EVSI).
- **TigerGraph MCP** runs on loopback beside Community Edition. The live agent's
  pattern scorers and prior-case similarity use the official
  [`tigergraph__run_installed_query`](https://github.com/tigergraph/tigergraph-mcp)
  tool through MCP client sessions. The live trace shows query names,
  parameters, and bounded previews of returned rows. The bounded evidence
  pack and audited writes use RESTPP. The MCP server exposes only that
  installed-query tool.
- **`gsql/`** holds the TigerGraph schema (15 vertex types, 29 edge types), loading jobs,
  26 installed queries including maintenance jobs, plus three graph algorithms (k-hop expand, mule fan-in, card-testing,
  ATO, out-of-region, structural similarity, evidence packs, community discovery),
  maintenance jobs, and `connect_and_verify.sh`, a graph census verifier.
- **`actions_api/`** has 14 policy-action endpoints with R1–R10 as executable guards;
  `fired_rules()` guarantees every rule cited in an answer actually fired.
- **`bench/`** is the benchmark runner over the 20-case pack; writes and validates the
  answer files (JSON Schema + semantic + ID-existence + rule-fired layers).
- **`graphrag/`** is a TF-IDF document index over the fraud policy, five typologies and
  regulatory references; gives the agent citable document evidence.
- **`scripts/`** covers teacher trace generation (Vertex AI), QLoRA SFT, deterministic
  eval gate, GGUF packaging, memory counterfactual, ablations.
- **`service/` + `ui/`** expose the live investigator workspace: a bounded TigerGraph
  neighborhood, step-by-step agent activity, recorded/live comparison, CSV/JSON
  imports for existing or new transactions, and an autonomous exam-period watchlist.

See [runtime architecture](docs/architecture.md) for data ownership, graph
read and write paths, import ordering, and deployment boundaries.

## Verified results

| Check | Result |
|---|---|
| Graph census (`gsql/connect_and_verify.sh`) | **14/14 PASS**: 590,742 source transactions plus five demo imports, 575,896 within-card NEXT edges, 5,565 closed cases, 27 fraud cases including imports, and 73,407 identity clusters |
| Installed GSQL queries | **26 queries + 3 graph algorithms** installed on the CPU VM |
| Decision model, October overlap holdout | **AUC 0.927, class-balanced Brier 0.109** on 295 closed cases where the historical file contains both outcomes; probabilities capped at 0.05–0.95. Hidden-case accuracy is unknown. |
| New-transaction fallback | Six-feature model: **AUC 0.808, balanced Brier 0.163** on the same October overlap. New 2026 transactions have no comparable labels. |
| Benchmark | **20/20** answer files in `cases/` validate against schema, semantics, IDs, and fired rules |
| Test suite | **87 passed** with the challenge CSVs present (`make test`); public CI runs the tests that need no private data |
| Graph write-back (task mandate) | **20/20** investigated cases upserted into the live graph via `q_record_case` + Evidence/Decision/AgentStep audit trail; `written_to_graph: true` with receipts. Uncertain cases with pending evidence remain open. |
| Ablations | no-graph flips **7** verdicts (mean \|Δp\| 0.163) · no-memory 3 · no-calibration 3 |
| Undocumented pattern | **cross-account shared-device ring**: Build/-qualified devices linking ≥3 cardholders on confirmed-fraud; permutation **p = 0.0005** (n=2000); demo exemplar SM-G935F (CC-2649/2971/2985/3035) |
| Autonomous watchlist | **186** unimported November–December transactions outside the case pack: bank risk <0.50, Build-qualified device shared with ≥3 other customers' earlier confirmed-fraud cases, and profile seen on ≤25 customers. One additional lead was imported as HHG-902. Candidates require review; these are not ground-truth fraud labels. |
| Model artifact | The Qwen3-14B QLoRA experiment and its failed raw-reply audit are documented; the large weights are omitted. The calibrated graph loop makes live decisions. |

**Submission audit (24 Sep):** An earlier answer set had 12 wrong transaction
channels and nine incorrect scored card links. The current 20 files were
regenerated from the rebuilt live graph and raw source fields using the
calibrated overlap model. `make answer-facts` reports zero channel, amount,
transaction-ID, or channel-pattern errors. The answer set contains 10 fraud,
six legitimate, and four uncertain verdicts. Hidden-key accuracy remains
unknown. See [the model and graph evidence](docs/evaluation.md).

### What we will not claim

The memory counterfactual is reported honestly: real device-level memory moves
`p` (mean |Δp| = 0.058, concentrated in the 7/20 cases with prior history) and
changes 1 verdict, but a volume-preserving permutation null produces 0–1 changes
too (p = 0.75). Device memory is real but decision-weak on this dataset. Details in
`eval/`.

The SFT artifact has a known limit. A schema-only generation check reported
20/20 parseable objects, but replaying the saved replies through the full
schema, semantic, ID, and policy checks gives 0/20 valid raw answers. The
training corpus also exposed outcome metadata in prompts and lost richer action
lists during normalization. We do not use that artifact to make decisions.
The calibrated graph loop remains the decision maker. `eval/sft/` holds the
audit evidence and sample replies.

The closed-case holdout is a selected investigation population, while the
challenge brief says half of its 20 cases are legitimate. The October metrics
therefore do not establish challenge accuracy. We audit live graph evidence
separately and do not claim a public leaderboard result or SOTA status.

## Run it

Download the task's [HHGOA_IEEE dataset](https://drive.google.com/drive/folders/1YDJUW1fiE7Jx8R9KqknC4IcsED9zll2A?usp=sharing)
and place `transactions.csv`, `identity.csv`, `closed_cases_history.csv`, and
`case_pack.csv` in `data/HHGOA_IEEE/`. The raw data is excluded from Git. Use
Python 3.11 and Node 20:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/prep_load_data.py
python -m graphrag.ingest

# No TigerGraph required: deterministic offline smoke run.
python -m bench.run --cases data/HHGOA_IEEE/case_pack.csv \
  --out /tmp/sumora-offline-answers --graph-agent --deterministic

python -m pytest bench/tests actions_api/tests -q
python scripts/verify_submission.py --require-source
cd ui && npm ci && npm run build
```

For the full graph-backed run, use the [CPU VM deployment guide](docs/deployment.md)
to start TigerGraph Community Edition 4.2.5, mount `prepped/`, and run
`bash gsql/connect_and_verify.sh load` once. Then:

```bash
cd /opt/sumora/current
bash gsql/connect_and_verify.sh verify
TG_HOST=127.0.0.1 TG_PORT=9000 /opt/sumora/venv/bin/python -m bench.run \
  --cases data/HHGOA_IEEE/case_pack.csv \
  --out /tmp/sumora-live-answers --graph-agent --live
```

The hosted dashboard is **http://34.66.141.236/**. Select a case to see its
recorded answer; **Graph** queries the live TigerGraph neighborhood, **Agent
activity** shows real steps during **Investigate live**, and **Replay recorded
trace** shows the saved run, including MCP result previews. When a verification
request has positive expected value, the agent leaves the case uncertain until
a real response arrives. **Watchlist** scans outside the scored pack, and
**Add data** accepts a 1–5 row CSV or JSON package in either mode:

- **Existing graph transaction:** `flagged_txn_id,trigger_type,trigger_text`
- **New transaction row:** `transaction_id,customer_id,card_id,ts,amount_usd,channel,risk_score,device_profile,trigger_text`

Both modes create a case in TigerGraph. New transaction assessments are marked
provisional because customer history may be sparse and the risk input is
user-supplied. Extra investigations live under VM run storage and, when
selected for the submission, `innovation/`; neither mode overwrites `cases/`.

The live URL, VM operation commands, cost, and model ownership are in
[`docs/deployment.md`](docs/deployment.md). The verified
TigerGraph choice for the submission form is **Community Edition**. The
calibrated gradient-boosting artifact in `eval/overlap_model/candidate.joblib`
runs inside the live agent process. Qwen3-14B remains an audited experiment and
does not make live decisions.

The VM was moved from Spot to on-demand during finalization after repeated
preemptions. The Spot estimate in the deployment guide is a cost target, not the
current on-demand run rate. The 20 committed answer files remain readable if
the live service is interrupted.

The public CI checks lint, tests that do not need the challenge CSVs, the
committed answer package, and the UI build. With the raw CSVs present,
`python scripts/verify_submission.py --require-source` also checks IDs and
source transaction facts. Reproduce model selection with
`python scripts/evaluate_overlap_model.py` and compare its
`eval/overlap_model/report.json` with the committed report.
That evaluation generates a local historical feature cache from the downloaded
CSV files; the cache and GraphRAG index are excluded from Git.

## Integrity rules honored

- All IDs in answer files exist in the dataset (validated, not promised).
- The public Kaggle IEEE-CIS files were never used to recover outcomes.
- Every rule citation is checked against the executable policy guards.
- Evaluation numbers come from the committed scripts, re-runnable end to end.
