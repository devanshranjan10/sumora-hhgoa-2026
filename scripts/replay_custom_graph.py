"""Restore imported cases after rebuilding the TigerGraph graph."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = "http://127.0.0.1:9000/graph/SumoraFraudGraph"
CUSTOM_PATH = Path(os.environ.get("SUMORA_CUSTOM_PATH", "/opt/sumora/run/custom-records.json"))


def request(url: str, payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json", "gsql-atomic-level": "atomic"}
    with urlopen(Request(url, data=body, headers=headers), timeout=30) as response:
        result = json.load(response)
    if result.get("error"):
        raise RuntimeError(result)
    return result


def value(item: object) -> dict:
    return {"value": item}


def transaction_dt(ts: str) -> float:
    moment = datetime.fromisoformat(ts)
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return (moment - datetime(2016, 6, 1)).total_seconds()


def main() -> None:
    records = json.loads(CUSTOM_PATH.read_text())
    for record in records:
        case = record["case"]
        txn_id = case["flagged_txn_id"]
        raw = txn_id.removeprefix("T")
        transaction = record.get("transaction")
        if transaction is None:
            edges = request(f"{ROOT}/edges/Transaction/{quote(raw)}/MADE_REV")["results"]
            if len(edges) != 1:
                raise RuntimeError(f"{txn_id}: expected exactly one card, got {len(edges)}")
            card_id = edges[0]["to_id"]
            case["card_id"] = card_id
            case["customer_id"] = card_id.split("-")[0]

        case_id = case["case_id"]
        vertices = {"FraudCase": {case_id: {"status": value("open"), "verdict": value("uncertain")}}}
        graph_edges = {"FraudCase": {case_id: {"INVESTIGATES": {"Transaction": {raw: {}}}}}}
        if transaction is not None:
            card_id = case["card_id"]
            customer_id = case["customer_id"]
            transaction["txn_dt"] = transaction_dt(transaction["ts"])
            vertices["Transaction"] = {raw: {
                "ts": value(transaction["ts"]), "amt": value(transaction["amount_usd"]),
                "txn_dt": value(transaction["txn_dt"]),
                "risk_score": value(transaction["risk_score"]),
                "channel": value(transaction["channel"]),
            }}
            vertices["Customer"] = {customer_id: {}}
            vertices["Card"] = {card_id: {}}
            graph_edges["Customer"] = {customer_id: {"OWNS": {"Card": {card_id: {}}}}}
            graph_edges["Card"] = {card_id: {"MADE": {"Transaction": {raw: {}}}}}
            previous = transaction.get("previous_txn_id")
            if previous:
                previous_raw = str(int(previous.removeprefix("T")))
                graph_edges["Transaction"] = {previous_raw: {
                    "NEXT": {"Transaction": {raw: {"weight": value(1)}}},
                }}
            device = transaction.get("device_profile")
            if device:
                vertices["DeviceProfile"] = {device: {"device_info": value(device)}}
                graph_edges.setdefault("Transaction", {})[raw] = {
                    "FROM_DEVICE": {"DeviceProfile": {device: {}}},
                }
        result = request(f"{ROOT}?ack=all&vertex_must_exist=true", {
            "vertices": vertices, "edges": graph_edges,
        })
        counts = result.get("results", [{}])[0]
        if counts.get("accepted_vertices", 0) < 1 or counts.get("accepted_edges", 0) < 1:
            raise RuntimeError(f"{case_id}: graph rejected import: {result}")
        print(f"{case_id}: {txn_id} on {case['card_id']}")

    temporary = CUSTOM_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, indent=2) + "\n")
    temporary.replace(CUSTOM_PATH)


if __name__ == "__main__":
    main()
