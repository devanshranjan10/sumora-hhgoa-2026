"""Verify each scored transaction is attached to its supplied card in TigerGraph.

Run on the graph host: python scripts/check_graph_card_links.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "data" / "HHGOA_IEEE" / "case_pack.csv"


def main() -> int:
    failures = []
    with httpx.Client(base_url="http://127.0.0.1:9000", timeout=20) as client:
        for row in csv.DictReader(PACK.open(newline="")):
            response = client.get(
                f"/graph/SumoraFraudGraph/edges/Transaction/"
                f"{row['flagged_txn_id'].removeprefix('T')}/MADE_REV"
            )
            response.raise_for_status()
            cards = sorted(edge["to_id"] for edge in response.json().get("results", []))
            if cards != [row["card_id"]]:
                failures.append(f"{row['case_id']}: {cards} != {[row['card_id']]}")
    for failure in failures:
        print(failure)
    print(f"{20 - len(failures)}/20 scored transaction-card links verified")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
