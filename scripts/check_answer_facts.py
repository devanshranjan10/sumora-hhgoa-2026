"""Check scored answers against source transaction facts.

Run: ./.venv/bin/python scripts/check_answer_facts.py [answers-directory]
This checks claims with an unambiguous source field; it is not a verdict oracle.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "HHGOA_IEEE"
CLAIM = re.compile(r"^Flagged transaction (T\d+) \(\$([\d,.]+), (online|card-present)\)$")


def check(directory: Path) -> list[str]:
    cases = {
        row["case_id"]: row
        for row in csv.DictReader((DATA / "case_pack.csv").open(newline=""))
    }
    wanted = {row["flagged_txn_id"].removeprefix("T").lstrip("0") for row in cases.values()}
    transactions = {}
    with (DATA / "transactions.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["TransactionID"].lstrip("0") in wanted:
                transactions[row["TransactionID"].lstrip("0")] = row
    errors = []
    expected_files = {f"{case_id}.json" for case_id in cases}
    actual_files = {p.name for p in directory.glob("*.json")}
    if actual_files != expected_files:
        errors.append(f"answer filenames differ: missing={sorted(expected_files - actual_files)}, extra={sorted(actual_files - expected_files)}")
    for case_id, case in cases.items():
        path = directory / f"{case_id}.json"
        if not path.exists():
            continue
        answer = json.loads(path.read_text())
        txn_id = f"T{int(case['flagged_txn_id'].removeprefix('T')):07d}"
        txn = transactions[txn_id.removeprefix("T").lstrip("0")]
        claim = answer["case"]["evidence"][0]["claim"]
        match = CLAIM.fullmatch(claim)
        if not match:
            errors.append(f"{case_id}: unparseable first evidence claim: {claim[:120]}")
            continue
        claimed_txn, claimed_amount, claimed_channel = match.groups()
        source_channel = "online" if txn["channel"] == "online" else "card-present"
        if claimed_txn != txn_id:
            errors.append(f"{case_id}: transaction {claimed_txn} != {txn_id}")
        if abs(float(claimed_amount.replace(",", "")) - float(txn["TransactionAmt"])) > 0.011:
            errors.append(f"{case_id}: amount ${claimed_amount} != ${txn['TransactionAmt']}")
        if claimed_channel != source_channel:
            errors.append(f"{case_id}: channel {claimed_channel} != {source_channel}")
        pattern = answer["case"]["pattern"]
        if source_channel == "card-present" and pattern.startswith("card_not_present"):
            errors.append(f"{case_id}: pattern {pattern} contradicts source channel")
    return errors


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cases"
    failures = check(folder)
    for failure in failures:
        print(failure)
    print(f"{len(failures)} source-fact errors")
    raise SystemExit(bool(failures))
