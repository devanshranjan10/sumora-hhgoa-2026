# CPU VM deployment

The live prototype runs on Google Compute Engine VM `sumora-cpu` in
`contral-6b0bd/us-central1-a` at **http://34.66.141.236/**. Nginx serves the
dashboard and proxies `/api/` to one Python worker. That worker keeps the
calibrated gradient-boosting model and CSV index in memory. TigerGraph
Community Edition 4.2.5 serves RESTPP on `127.0.0.1:9000`; the official
TigerGraph MCP server serves the agent's pattern scorers and prior-case
similarity queries on
`127.0.0.1:8001/mcp/`. Both bind to loopback, so the public VM exposes only
HTTP and SSH.

| Component | Location | Purpose |
|---|---|---|
| Dashboard | Nginx, `/opt/sumora/current/ui/dist` | Case graph, live agent steps, 20 recorded answers, imports, and watchlist |
| Agent API | `sumora-agent.service`, `/opt/sumora/current/service/app.py` | Runs the LangGraph loop, imports cases and transactions, and serves the watchlist |
| Decision model | `/opt/sumora/current/eval/overlap_model/candidate.joblib` | CPU-only calibrated gradient boosting over supplied raw fields, with a six-feature fallback for new rows |
| TigerGraph | Docker container `tigergraph`, image `tigergraph/community:4.2.5` | Graph queries and audited case write-back |
| TigerGraph MCP | `sumora-mcp.service`, official `tigergraph-mcp==1.0.3` | Exposes installed pattern and similarity queries to the agent over MCP; only `tigergraph__run_installed_query` is enabled |
| Raw data | `/opt/sumora/data/HHGOA_IEEE/raw` | The agent's source fields and dataset index; `current/data/HHGOA_IEEE` points here |
| Prepared graph data | `/opt/sumora/data/HHGOA_IEEE/prepped` | Derived card keys and `NEXT` edges, mounted read-only into TigerGraph |

The Qwen3-14B QLoRA/GGUF artifact is an experiment. Saved raw replies fail the
full answer contract, and its 8.4 GB file would compete with TigerGraph for
RAM on this VM. It does not make the fraud decisions. The deployed answer model
is the calibrated gradient-boosting artifact; `EchoProvider` supplies deterministic prose.

## Cost and availability

The VM is `e2-highmem-2` (2 vCPU, 16 GiB RAM) with a 100 GB standard
persistent disk and one reserved IPv4 address. We first used **Spot** at an
estimated **$47.25/month** ($39.60 compute + $4 disk + $3.65 IP). It was
preempted twice during final verification, and its zone then had no Spot
capacity. We switched the same disk and IP to **standard on-demand** for the
submission window. At the published $0.09039966/hour compute rate, 730 hours
cost about **$73.64/month** with disk and IP, before egress and tax. This
exceeds the original $30–50/month operating target but keeps the live demo
available. See [Compute Engine pricing](https://cloud.google.com/products/compute/pricing/general-purpose),
[Spot provisioning](https://docs.cloud.google.com/compute/docs/instances/create-use-spot),
[disk pricing](https://cloud.google.com/compute/disks-image-pricing), and
[external IP pricing](https://cloud.google.com/vpc/pricing).

The boot disk, reserved IP, Docker data bind mounts, and enabled systemd
services survived the Spot preemptions. Check the live endpoint before
recording or submitting the demo. Standard capacity is the current setting;
switching back to Spot later would restore the lower estimate but allow new
interruptions.

## Operate the live deployment

```bash
gcloud compute instances describe sumora-cpu --project=contral-6b0bd \
  --zone=us-central1-a --format='value(status)'
gcloud compute ssh sumora-cpu --project=contral-6b0bd --zone=us-central1-a \
  --command='sudo systemctl is-active sumora-agent sumora-mcp nginx docker; sudo docker ps --filter name=tigergraph'
curl http://34.66.141.236/api/health
```

`/api/health` checks that `q_case_feature_vector` returns a populated Louvain
community, the MCP listener is up, and the model is loaded. The agent fails
the investigation if its MCP query fails instead of silently dropping that
evidence. When all three are ready, run a case from the dashboard or call:

```bash
curl -X POST http://34.66.141.236/api/cases/HHG-001/investigate
```

The API accepts the 20 case-pack IDs plus up to 100 imported cases (`HHG-900`–`HHG-999`).
`POST /api/import` accepts up to five existing-graph cases or new transaction rows
per request; it writes their vertices and edges before adding them to the in-memory
index, then persists the definitions in `/opt/sumora/run/custom-records.json`.
New rows receive a seconds-from-dataset-start graph time and a `NEXT` edge to
the preceding transaction on their card. A row on an existing card must be
later than that card's current last transaction; rows in one package must be
in card order.
`GET /api/watchlist` returns a bounded, time-aware scan of low-risk exam-period
transactions linked to earlier confirmed fraud through a Build-qualified device
profile. It is an alert queue, not a fraud classifier.

The service serializes investigations to protect the 2-vCPU VM, Nginx rate-limits
investigation and import requests, and the answer passes
the same schema/semantic/ID/policy validator as `bench.run`. Live answers and
traces are stored separately under `/opt/sumora/run/`; the submitted `cases/`
files are not changed by a demo run.

For a graph census, run this **read-only** verifier on the VM:

```bash
cd /opt/sumora/current
bash gsql/connect_and_verify.sh verify
```

Keep the raw and prepared directories separate. Filling missing raw fields with
zero before model inference changes its probabilities. The graph's persistent
data is under `/opt/sumora/tigergraph-data` and logs
under `/opt/sumora/tigergraph-log`. The service and Nginx configurations are in
`deploy/`. The MCP service reads TigerGraph credentials from the root-owned
`/etc/sumora/tigergraph-mcp.env` (mode 0600); keep that file out of Git and
start `sumora-mcp.service` before `sumora-agent.service`. On a fresh Linux host
with Docker, at least 16 GiB RAM, and the
challenge CSVs prepped by `scripts/prep_load_data.py`, the graph can be created
with named volumes (which seed the image's initial TigerGraph configuration):

```bash
sudo docker volume create sumora_tg_data
sudo docker volume create sumora_tg_log
sudo docker run -d --name tigergraph --restart unless-stopped \
  --ulimit nofile=1000000:1000000 \
  -p 127.0.0.1:9000:9000 -p 127.0.0.1:14240:14240 \
  -v "$PWD/data/HHGOA_IEEE/prepped:/home/tigergraph/data_in/HHGOA_IEEE/prepped:ro" \
  -v sumora_tg_data:/home/tigergraph/tigergraph/data \
  -v sumora_tg_log:/home/tigergraph/tigergraph/log \
  tigergraph/community:4.2.5
bash gsql/connect_and_verify.sh load
bash gsql/connect_and_verify.sh verify
```

The load step creates the schema, imports data, computes graph features, and
installs queries; it is not needed when restarting this VM. It can take tens
of minutes on a 2-vCPU host because TigerGraph compiles each GSQL query.

## Submission setting

Select **Community Edition** in the TigerGraph form. Savanna is not deployed.
The earlier A100 development VM was backed up and stopped. The CPU VM is the
only live prototype host. Its current graph census passes 14/14 checks with
the imported examples present.
