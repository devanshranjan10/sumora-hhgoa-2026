"""Serve bounded, live TigerGraph investigations to the demo dashboard."""

from __future__ import annotations

import asyncio
import os
import json
import uuid
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import httpx
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from actions_api.personas import ScriptedPersonaResponder
from agent.runner import GraphAgentRunner
from agent.source_features import SourceFeatureStore
from bench.agent import CaseContext
from bench.dataset import DATA_DIR, load_dataset, norm_txn_id, txn_id_variants
from bench.instrument import CaseMetrics
from bench.run import DEFAULT_FIXTURES, DETERMINISTIC_SEED, load_cases
from bench.writer import write_answer
from service.watchlist import build_watchlist


TRACE_DIR = Path(os.environ.get("SUMORA_TRACE_DIR", "/opt/sumora/run/traces"))
ANSWER_DIR = Path(os.environ.get("SUMORA_ANSWER_DIR", "/opt/sumora/run/answers"))
CACHE_DIR = Path(os.environ.get("SUMORA_CACHE_DIR", "/opt/sumora/run/cache"))
CUSTOM_PATH = Path(os.environ.get("SUMORA_CUSTOM_PATH", "/opt/sumora/run/custom-records.json"))
DATASET_START = datetime(2016, 6, 1)


def _transaction_dt(ts: str) -> float:
    moment = datetime.fromisoformat(ts)
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return (moment - DATASET_START).total_seconds()


def _load_or_build(cache: Path, sources: list[Path], build):
    if cache.exists() and all(cache.stat().st_mtime_ns >= source.stat().st_mtime_ns for source in sources):
        try:
            return joblib.load(cache)
        except (EOFError, OSError, ValueError):
            pass
    value = build()
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp")
    joblib.dump(value, temporary)
    temporary.replace(cache)
    return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cases = {
        case.case_id: case for case in load_cases(DATA_DIR / "case_pack.csv")
    }
    app.state.dataset = _load_or_build(
        CACHE_DIR / "dataset-index.joblib",
        list(DATA_DIR.glob("*.csv")),
        lambda: load_dataset(DATA_DIR),
    )
    app.state.custom_records = json.loads(CUSTOM_PATH.read_text()) if CUSTOM_PATH.exists() else []
    for record in app.state.custom_records:
        _register_custom(record)
    app.state.watchlist = build_watchlist(
        app.state.dataset,
        {norm_txn_id(case.flagged_txn_id) for case in app.state.cases.values()},
    )
    fixtures = str(DEFAULT_FIXTURES) if DEFAULT_FIXTURES.exists() else None
    app.state.runner = GraphAgentRunner(
        persona=ScriptedPersonaResponder(fixtures, seed=DETERMINISTIC_SEED),
        seed=DETERMINISTIC_SEED,
        mode="live",
    )
    model = app.state.runner._calibrator(app.state.dataset)
    model.source_store = SourceFeatureStore(DATA_DIR)
    app.state.run_lock = Lock()
    app.state.jobs_lock = Lock()
    app.state.jobs = {}
    app.state.import_lock = Lock()
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    ANSWER_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Sumora live investigation", lifespan=lifespan)


def _register_custom(record: dict) -> None:
    case = CaseContext(**record["case"])
    app.state.cases[case.case_id] = case
    idx = app.state.dataset
    t = norm_txn_id(case.flagged_txn_id)
    idx.txn_card[t] = case.card_id
    idx.card_customer[case.card_id] = case.customer_id
    idx.customer_cards.setdefault(case.customer_id, [])
    if case.card_id not in idx.customer_cards[case.customer_id]:
        idx.customer_cards[case.customer_id].append(case.card_id)
    idx.card_txns.setdefault(case.card_id, [])
    if t not in idx.card_txns[case.card_id]:
        idx.card_txns[case.card_id].append(t)
    txn = record.get("transaction")
    if txn:
        idx.txn_ids.add(t)
        idx.txn_amount[t] = txn["amount_usd"]
        idx.txn_ts[t] = txn["ts"]
        idx.txn_dt[t] = txn.get("txn_dt", _transaction_dt(txn["ts"]))
        idx.txn_risk[t] = txn["risk_score"]
        idx.txn_online[t] = txn["channel"] == "online"
        idx.txn_customer[t] = case.customer_id
        idx.customer_txns.setdefault(case.customer_id, []).append(t)
        if txn.get("device_profile"):
            idx.txn_device[t] = txn["device_profile"]
            idx.device_profiles.add(txn["device_profile"])


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            ping = await client.get("http://127.0.0.1:14240/api/ping")
            ping.raise_for_status()
            query = await client.post(
                "http://127.0.0.1:9000/query/SumoraFraudGraph/q_case_feature_vector",
                json={"case_id": "HHG-001"},
            )
            query.raise_for_status()
            result = query.json()
            graph_ready = (
                ping.json().get("message") == "pong"
                and not result.get("error", True)
                and result["results"][0]["flagged_community"] > 0
            )
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        graph_ready = False
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", 8001), 2)
        writer.close()
        await writer.wait_closed()
        mcp_ready = True
    except (OSError, TimeoutError):
        mcp_ready = False
    return {
        "graph_ready": graph_ready,
        "mcp_ready": mcp_ready,
        "model_ready": hasattr(app.state, "runner"),
        "case_count": len(app.state.cases),
        "deployment": "TigerGraph Community Edition + calibrated gradient boosting, CPU",
    }


def _investigate(case_id: str):
    if not app.state.run_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another investigation is running")
    try:
        metrics = CaseMetrics()
        metrics.start()
        result = app.state.runner.investigate(
            app.state.cases[case_id], app.state.dataset, metrics
        )
        answer = result.answer
        answer["latency_s"] = metrics.latency_s
        answer["tool_calls"] = metrics.tool_calls
        answer["tokens"] = metrics.tokens
        write_answer(
            answer, ANSWER_DIR, idx=app.state.dataset,
            case_state=result.case_state,
        )
        trace = (TRACE_DIR / f"{case_id}.trace.json").read_text()
        return {"answer": answer, "trace": json.loads(trace)}
    finally:
        app.state.run_lock.release()


@app.post("/api/cases/{case_id}/investigate")
async def investigate(case_id: str):
    if case_id not in app.state.cases:
        raise HTTPException(status_code=404, detail="Unknown case")
    result = await health()
    if not result["graph_ready"] or not result["mcp_ready"]:
        raise HTTPException(status_code=503, detail="TigerGraph or its MCP server is unavailable")
    return await run_in_threadpool(_investigate, case_id)


@app.get("/api/catalog")
def catalog():
    idx = app.state.dataset
    return {
        "submission_cases": len(app.state.cases) - len(app.state.custom_records),
        "imported_cases": len(app.state.custom_records),
        "transactions": len(idx.txn_ids),
        "closed_cases": len(idx.closed_case_ids),
        "customers": len(idx.customer_txns),
        "cards": len(idx.card_customer),
        "device_profiles": len(idx.device_profiles),
    }


@app.get("/api/watchlist")
def watchlist():
    return app.state.watchlist


@app.get("/api/cases")
@app.get("/api/cases/")
def list_cases():
    return {"case_ids": sorted(app.state.cases),
            "imported_case_ids": [record["case"]["case_id"] for record in app.state.custom_records],
            "new_transaction_case_ids": [record["case"]["case_id"] for record in app.state.custom_records
                                         if record.get("transaction")]}


def _value(value):
    return {"value": value}


async def _graph_identity(client: httpx.AsyncClient, transaction_id: str) -> tuple[str, str]:
    _, raw = txn_id_variants(transaction_id)
    card_response = await client.get(
        f"http://127.0.0.1:9000/graph/SumoraFraudGraph/edges/Transaction/{raw}/MADE_REV"
    )
    card_response.raise_for_status()
    cards = card_response.json().get("results", [])
    if len(cards) != 1:
        raise HTTPException(status_code=422, detail="The transaction must have exactly one card in TigerGraph")
    card_id = cards[0]["to_id"]
    customer_response = await client.get(
        f"http://127.0.0.1:9000/graph/SumoraFraudGraph/edges/Card/{card_id}/OWNS_REV"
    )
    customer_response.raise_for_status()
    customers = customer_response.json().get("results", [])
    if len(customers) != 1:
        raise HTTPException(status_code=422, detail="The card must have exactly one customer in TigerGraph")
    return card_id, customers[0]["to_id"]


@app.post("/api/import")
async def import_package(payload: dict):
    """Import at most five new case definitions or new transaction rows."""
    rows = payload.get("rows")
    kind = payload.get("kind")
    if kind not in ("case", "transaction") or not isinstance(rows, list) or not 1 <= len(rows) <= 5:
        raise HTTPException(status_code=422, detail="Choose case or transaction and supply 1–5 rows")
    if not app.state.import_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another import is in progress")
    try:
        if len(app.state.custom_records) + len(rows) > 100:
            raise HTTPException(status_code=422, detail="This demo supports at most 100 imported cases")
        prepared = []
        used = set(app.state.cases)
        new_transaction_ids = set()
        pending_card_latest = {}
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        async with httpx.AsyncClient(timeout=15) as client:
            for row in rows:
                if not isinstance(row, dict):
                    raise HTTPException(status_code=422, detail="Every row must be an object")
                case_id = next((f"HHG-{number:03d}" for number in range(900, 1000)
                                if f"HHG-{number:03d}" not in used), None)
                if case_id is None:
                    raise HTTPException(status_code=422, detail="No import case IDs remain")
                used.add(case_id)
                transaction_id = norm_txn_id(row.get("flagged_txn_id") or row.get("transaction_id") or "")
                if not re.fullmatch(r"T\d{7}", transaction_id):
                    raise HTTPException(status_code=422, detail="Transaction ID must use T plus seven digits")
                trigger_type = str(row.get("trigger_type") or "analyst_request")
                if trigger_type not in ("risk_score", "customer_report", "analyst_request"):
                    raise HTTPException(status_code=422, detail="Unsupported trigger type")
                trigger_text = str(row.get("trigger_text") or "Imported for investigation").strip()[:300]
                if not trigger_text:
                    raise HTTPException(status_code=422, detail="Trigger text is required")
                transaction = None
                if kind == "case":
                    if transaction_id not in app.state.dataset.txn_ids:
                        raise HTTPException(status_code=422, detail=f"{transaction_id} is not in the loaded graph dataset")
                    card_id, customer_id = await _graph_identity(client, transaction_id)
                    if row.get("card_id") and row["card_id"] != card_id:
                        raise HTTPException(status_code=422, detail=f"{transaction_id} card does not match TigerGraph")
                    if row.get("customer_id") and row["customer_id"] != customer_id:
                        raise HTTPException(status_code=422, detail=f"{transaction_id} customer does not match TigerGraph")
                    risk = app.state.dataset.txn_risk[transaction_id]
                else:
                    if (transaction_id in app.state.dataset.txn_ids or transaction_id in new_transaction_ids
                            or not transaction_id.startswith("T9")):
                        raise HTTPException(status_code=422, detail="New transactions need an unused T9xxxxxx ID")
                    new_transaction_ids.add(transaction_id)
                    card_id = str(row.get("card_id") or "")
                    customer_id = str(row.get("customer_id") or "")
                    if not re.fullmatch(r"C\d{5}-K\d", card_id) or not re.fullmatch(r"C\d{5}", customer_id):
                        raise HTTPException(status_code=422, detail="Use customer C##### and card C#####-K#")
                    if app.state.dataset.card_customer.get(card_id, customer_id) != customer_id:
                        raise HTTPException(status_code=422, detail="Card belongs to a different customer")
                    try:
                        amount = float(row["amount_usd"])
                        risk = float(row["risk_score"])
                    except (KeyError, TypeError, ValueError):
                        raise HTTPException(status_code=422, detail="amount_usd and risk_score must be numeric")
                    if not 0 < amount <= 1000000 or not 0 <= risk <= 1:
                        raise HTTPException(status_code=422, detail="Amount or risk score is outside its allowed range")
                    channel = str(row.get("channel") or "")
                    if channel not in ("online", "in_person"):
                        raise HTTPException(status_code=422, detail="Channel must be online or in_person")
                    ts = str(row.get("ts") or "")
                    try:
                        txn_dt = _transaction_dt(ts)
                    except ValueError:
                        raise HTTPException(status_code=422, detail="ts must be an ISO date-time")
                    prior = pending_card_latest.get(card_id)
                    if prior is None:
                        prior = max(
                            app.state.dataset.card_txns.get(card_id, []),
                            key=lambda tid: app.state.dataset.txn_dt[tid], default=None,
                        )
                        prior_dt = app.state.dataset.txn_dt[prior] if prior else float("-inf")
                    else:
                        prior_dt = prior[1]
                        prior = prior[0]
                    if txn_dt <= prior_dt:
                        raise HTTPException(
                            status_code=422,
                            detail="New transactions must follow the latest transaction on their card",
                        )
                    pending_card_latest[card_id] = (transaction_id, txn_dt)
                    device = str(row.get("device_profile") or "").strip()[:150]
                    if channel == "online" and not device:
                        raise HTTPException(status_code=422, detail="Online transactions need a device_profile")
                    transaction = {"amount_usd": amount, "risk_score": risk, "channel": channel,
                                   "ts": ts, "txn_dt": txn_dt, "device_profile": device,
                                   "previous_txn_id": prior}
                context = {
                    "case_id": case_id, "opened_at": now, "trigger_type": trigger_type,
                    "trigger_text": trigger_text, "flagged_txn_id": transaction_id,
                    "card_id": card_id, "customer_id": customer_id, "risk_score": risk,
                }
                vertices = {"FraudCase": {case_id: {"status": _value("open"), "verdict": _value("uncertain")}}}
                edges = {"FraudCase": {case_id: {"INVESTIGATES": {
                    "Transaction": {txn_id_variants(transaction_id)[1]: {}}}}}}
                if transaction:
                    raw = txn_id_variants(transaction_id)[1]
                    vertices["Transaction"] = {raw: {
                        "ts": _value(transaction["ts"]), "amt": _value(transaction["amount_usd"]),
                        "txn_dt": _value(transaction["txn_dt"]),
                        "risk_score": _value(risk), "channel": _value(transaction["channel"]),
                    }}
                    vertices["Customer"] = {customer_id: {}}
                    vertices["Card"] = {card_id: {}}
                    edges["Customer"] = {customer_id: {"OWNS": {"Card": {card_id: {}}}}}
                    edges["Card"] = {card_id: {"MADE": {"Transaction": {raw: {}}}}}
                    if prior:
                        previous_raw = txn_id_variants(prior)[1]
                        edges["Transaction"] = {previous_raw: {
                            "NEXT": {"Transaction": {raw: {"weight": _value(1)}}},
                        }}
                    if transaction["device_profile"]:
                        device = transaction["device_profile"]
                        vertices["DeviceProfile"] = {device: {"device_info": _value(device)}}
                        edges.setdefault("Transaction", {})[raw] = {
                            "FROM_DEVICE": {"DeviceProfile": {device: {}}},
                        }
                prepared.append(({"case": context, "transaction": transaction},
                                 {"vertices": vertices, "edges": edges}))
            # Validate every row before any graph write. Each row then commits atomically.
            for record, graph_data in prepared:
                response = await client.post(
                    "http://127.0.0.1:9000/graph/SumoraFraudGraph?ack=all&vertex_must_exist=true",
                    json=graph_data, headers={"gsql-atomic-level": "atomic"},
                )
                response.raise_for_status()
                result = response.json()
                counts = result.get("results", [{}])[0]
                if result.get("error") or counts.get("accepted_vertices", 0) < 1 or counts.get("accepted_edges", 0) < 1:
                    raise HTTPException(status_code=502, detail="TigerGraph rejected the import")
                app.state.custom_records.append(record)
                _register_custom(record)
                CUSTOM_PATH.parent.mkdir(parents=True, exist_ok=True)
                temporary = CUSTOM_PATH.with_suffix(".tmp")
                temporary.write_text(json.dumps(app.state.custom_records, indent=2))
                temporary.replace(CUSTOM_PATH)
        return {"case_ids": [record["case"]["case_id"] for record, _ in prepared],
                "kind": kind, "count": len(prepared)}
    finally:
        app.state.import_lock.release()


@app.get("/api/cases/{case_id}/graph")
async def case_graph(case_id: str):
    if case_id not in app.state.cases:
        raise HTTPException(status_code=404, detail="Unknown case")
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            "http://127.0.0.1:9000/query/SumoraFraudGraph/q_evidence_pack",
            json={"case_id": case_id},
        )
        response.raise_for_status()
    payload = response.json()
    if payload.get("error") or not payload.get("results"):
        raise HTTPException(status_code=502, detail="Graph query failed")
    raw = payload["results"][0]
    context = app.state.cases[case_id]
    _, anchor = txn_id_variants(context.flagged_txn_id)
    txn_info = {}
    for record in raw.get("transactions", []):
        parts = record.split("|", 5)
        if len(parts) == 6:
            txn_info[parts[0]] = {"amount_usd": parts[1], "bank_risk_score": parts[2],
                                  "channel": parts[4], "time": parts[5]}
    parsed = []
    for record in raw.get("edges", []):
        parts = record.split("|", 2)
        if len(parts) == 3:
            parsed.append(tuple(parts))
    direct = [edge for edge in parsed if edge[0] == anchor]
    direct_targets = {edge[2] for edge in direct}
    neighbors = [edge for edge in parsed if edge[2] in direct_targets and edge[0] != anchor and edge[1] in ("MADE", "FROM_DEVICE")]
    neighbors.sort(key=lambda edge: float(txn_info.get(edge[0], {}).get("bank_risk_score", 0)), reverse=True)
    selected = direct[:8] + neighbors[:8]
    card_ids = {edge[2] for edge in selected if edge[1] == "MADE"}
    selected += [edge for edge in parsed if edge[1] == "IN_CLUSTER" and edge[0] in card_ids][:3]
    selected = selected[:28]
    node_types = {anchor: "Transaction", case_id: "FraudCase"}
    for source, relation, target in selected:
        if relation != "IN_CLUSTER":
            node_types[source] = "Transaction"
        node_types[target] = {
            "MADE": "Card", "FROM_DEVICE": "DeviceProfile", "EMAIL": "EmailAddress",
            "BILLED_IN": "BillingRegion", "IN_CLUSTER": "IdentityCluster",
        }[relation]
    nodes = []
    for node_id, node_type in node_types.items():
        label = (f"T{int(node_id):07d}" if node_type == "Transaction" and node_id.isdigit()
                 else node_id)
        nodes.append({"id": node_id, "label": label, "type": node_type,
                      "focus": node_id == anchor,
                      "details": txn_info.get(node_id, {})})
    edges = [{"source": case_id, "target": anchor, "type": "INVESTIGATES"}]
    edges += [{"source": source, "target": target, "type": relation}
              for source, relation, target in selected]
    return {"case_id": case_id, "nodes": nodes, "edges": edges,
            "txn_candidates": raw.get("txn_candidates", 0),
            "edge_candidates": raw.get("edge_candidates", 0)}


@app.get("/api/cases/{case_id}/latest")
def latest_case_run(case_id: str):
    if case_id not in app.state.cases:
        raise HTTPException(status_code=404, detail="Unknown case")
    answer_path = ANSWER_DIR / f"{case_id}.json"
    trace_path = TRACE_DIR / f"{case_id}.trace.json"
    if not answer_path.exists() or not trace_path.exists():
        raise HTTPException(status_code=404, detail="This case has not been investigated yet")
    return {"answer": json.loads(answer_path.read_text()),
            "trace": json.loads(trace_path.read_text())}


def _run_job(case_id: str, job_id: str):
    def on_step(step):
        with app.state.jobs_lock:
            app.state.jobs[job_id]["steps"].append(step)
    try:
        metrics = CaseMetrics()
        metrics.start()
        result = app.state.runner.investigate(
            app.state.cases[case_id], app.state.dataset, metrics, on_step=on_step
        )
        answer = result.answer
        answer["latency_s"] = metrics.latency_s
        answer["tool_calls"] = metrics.tool_calls
        answer["tokens"] = metrics.tokens
        write_answer(answer, ANSWER_DIR, idx=app.state.dataset, case_state=result.case_state)
        trace = json.loads((TRACE_DIR / f"{case_id}.trace.json").read_text())
        with app.state.jobs_lock:
            app.state.jobs[job_id].update(status="complete", answer=answer, trace=trace)
    except Exception as exc:
        with app.state.jobs_lock:
            app.state.jobs[job_id].update(status="failed", error=str(exc))
    finally:
        app.state.run_lock.release()


@app.post("/api/cases/{case_id}/runs")
async def start_run(case_id: str):
    from threading import Thread
    if case_id not in app.state.cases:
        raise HTTPException(status_code=404, detail="Unknown case")
    readiness = await health()
    if not readiness["graph_ready"] or not readiness["mcp_ready"]:
        raise HTTPException(status_code=503, detail="TigerGraph or its MCP server is unavailable")
    if not app.state.run_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another investigation is running")
    job_id = uuid.uuid4().hex
    with app.state.jobs_lock:
        if len(app.state.jobs) >= 20:
            app.state.jobs.pop(next(iter(app.state.jobs)))
        app.state.jobs[job_id] = {"case_id": case_id, "status": "running", "steps": []}
    Thread(target=_run_job, args=(case_id, job_id), daemon=True).start()
    return {"run_id": job_id}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    with app.state.jobs_lock:
        job = app.state.jobs.get(run_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown run")
        return dict(job)
