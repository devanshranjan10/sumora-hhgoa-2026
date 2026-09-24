# TigerGraph graph

`SumoraFraudGraph` runs on TigerGraph Community Edition 4.2.5. The schema has
15 vertex types and 29 edge types, including reverse edges. The load starts
with 590,742 source transactions and 5,565 closed cases. Demo imports add
transactions and fraud cases after the base load.

## Load the challenge data

Place the four raw challenge CSVs in `data/HHGOA_IEEE/`. Keep that directory
out of Git. Prepare the graph input, then load it into a TigerGraph container
that mounts `data/HHGOA_IEEE/prepped/` at
`/home/tigergraph/data_in/HHGOA_IEEE/prepped`:

```bash
python scripts/prep_load_data.py
bash gsql/connect_and_verify.sh load
bash gsql/connect_and_verify.sh verify
```

`connect_and_verify.sh load` creates the schema, runs eight loading jobs,
installs the GSQL queries, populates graph features, and runs the algorithm
suite. It also loads 575,892 chronological `NEXT` edges within cards. The
script compiles queries on the two-vCPU VM, so a first load can take tens of
minutes. Do not run `load` for an ordinary service restart.

The verifier prints 14 checks and ends with `RESULT: 14 passed, 0 failed` on
the deployed graph. Its transaction and case thresholds allow later imports.
`scripts/check_graph_card_links.py` separately verifies the 20 scored
transaction-to-card links.

## Files and query use

| File | Role |
|---|---|
| `schema.gsql` | Vertex and edge types |
| `loading.gsql` | Eight loading jobs for source rows, `NEXT` edges, identity edges, closed cases, and the case pack |
| `jobs.gsql` | Case labels, identity clusters, pattern priors, and algorithm preparation |
| `algorithms.gsql` | Louvain and degree computation |
| `queries.gsql` | Evidence packs, structural similarity, case write-back, and other investigation queries |
| `pattern_scorers.gsql` | Fraud-pattern scorers |
| `discovery.gsql` | Community and fingerprint queries |
| `install_algorithms.sh` | Install the three graph algorithms used by the load |

The live agent calls selected installed read queries through the official
[TigerGraph MCP server](https://github.com/tigergraph/tigergraph-mcp).
Bounded evidence reads and audited writes use RESTPP. `agent/tools.py` names
the MCP read queries; `agent/mcp_graph.py` decodes their responses. The MCP
server and RESTPP bind to loopback on the CPU VM.

## GSQL implementation notes

The script copies whole `.gsql` files into the container and runs `gsql -f`.
Piping multiline definitions into the CLI can split a query before it closes.
Preparation materializes card, email, and cluster keys because the TigerGraph
4.2 loader cannot concatenate strings in a `LOAD` expression. The graph uses
`FraudCase` because `CASE` is a GSQL keyword. Transaction time is stored as
`DOUBLE` seconds from the dataset start for chronological queries.

See [the deployment guide](../docs/deployment.md) for the container command,
data mounts, services, and health checks.
