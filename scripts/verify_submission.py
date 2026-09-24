"""Verify the committed answer package, including source IDs when available."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.dataset import DATA_DIR, load_dataset
from bench.validate import validate_answer, validate_schema, validate_semantics
from scripts.check_answer_facts import check as check_source_facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-source", action="store_true",
                        help="fail if the challenge CSVs are unavailable")
    args = parser.parse_args()
    source_available = (DATA_DIR / "transactions.csv").is_file()
    if args.require_source and not source_available:
        parser.error("place the raw challenge CSVs in data/HHGOA_IEEE first")

    case_dir = ROOT / "cases"
    files = sorted(case_dir.glob("*.json"))
    expected = [f"HHG-{number:03d}.json" for number in range(1, 21)]
    if [path.name for path in files] != expected:
        print("cases/ must contain exactly HHG-001.json through HHG-020.json")
        return 1

    idx = load_dataset(DATA_DIR) if source_available else None
    failures = []
    for path in files:
        answer = json.loads(path.read_text())
        errors = validate_answer(answer, idx=idx) if idx else validate_schema(answer)
        if idx is None and not errors:
            errors.extend(validate_semantics(answer))
        if errors:
            failures.append((path.name, errors))
            continue
        if answer["case_id"] != path.stem:
            errors.append("case_id does not match filename")
        if not answer["case"]["written_to_graph"]:
            errors.append("write-back receipt is false")
        recorded = ROOT / "ui/public/data/answers" / path.name
        if not recorded.is_file() or recorded.read_bytes() != path.read_bytes():
            errors.append("dashboard answer differs from cases/")
        trace = ROOT / "ui/public/data/traces" / f"{path.stem}.trace.json"
        if not trace.is_file():
            errors.append("dashboard trace is missing")
        if errors:
            failures.append((path.name, errors))

    if source_available:
        failures.extend(("source facts", [error]) for error in check_source_facts(case_dir))

    for name, errors in failures:
        print(f"{name}: {'; '.join(errors)}")
    if failures:
        return 1
    level = "schema, semantics, IDs, rules, source facts" if idx else "schema and semantics"
    print(f"20/20 answers: {level}, write-back, dashboard parity, and traces verified")
    if not idx:
        print("Source ID and fact checks require the untracked challenge CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
