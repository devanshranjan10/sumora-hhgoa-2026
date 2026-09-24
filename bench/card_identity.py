"""Resolve card identities from the supplied case labels and card attributes.

`customer_id` identifies an issuer-derived account group, not a card. A missing
`card2` field does not encode K1/K2. Known case links are authoritative; an
unlabeled card fingerprint gets a stable internal U-id rather than a guessed
K-id that could merge unrelated cards.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

CARD_FIELDS = tuple(f"card{i}" for i in range(1, 7))


def fingerprint(row: dict[str, str]) -> tuple[str, ...]:
    return (row.get("customer_id", "").strip(),) + tuple(
        (row.get(name) or "").strip() for name in CARD_FIELDS
    )


def authoritative_card_links(data_dir: Path) -> dict[str, str]:
    links: dict[str, str] = {}

    def add(txn_id: str, card_id: str) -> None:
        previous = links.setdefault(txn_id, card_id)
        if previous != card_id:
            raise ValueError(f"conflicting card links for transaction {txn_id}")

    with (data_dir / "closed_cases_history.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            for txn_id in row["txn_ids"].split("|"):
                if txn_id:
                    add(txn_id.removeprefix("T"), row["card_id"])
    case_pack = data_dir / "case_pack.csv"
    if case_pack.exists():
        with case_pack.open(newline="") as handle:
            for row in csv.DictReader(handle):
                add(row["flagged_txn_id"].removeprefix("T"), row["card_id"])
    return links


def learn_card_fingerprints(data_dir: Path) -> tuple[dict[tuple[str, ...], str], dict[str, str]]:
    links = authoritative_card_links(data_dir)
    mapping: dict[tuple[str, ...], str] = {}
    found: set[str] = set()
    with (data_dir / "transactions.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            txn_id = row["TransactionID"]
            card_id = links.get(txn_id)
            if card_id is None:
                continue
            found.add(txn_id)
            key = fingerprint(row)
            previous = mapping.setdefault(key, card_id)
            if previous != card_id:
                raise ValueError(f"card fingerprint maps to both {previous} and {card_id}")
    missing = links.keys() - found
    if missing:
        raise ValueError(f"authoritative transactions missing from source: {sorted(missing)[:5]}")
    return mapping, links


def resolve_card(row: dict[str, str], mapping: dict[tuple[str, ...], str],
                 links: dict[str, str]) -> str:
    txn_id = row["TransactionID"].removeprefix("T")
    if txn_id in links:
        return links[txn_id]
    key = fingerprint(row)
    if key in mapping:
        return mapping[key]
    digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]
    return f"{key[0]}-U{digest}"
