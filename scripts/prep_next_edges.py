"""Materialize chronological next-transaction edges within each resolved card."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/HHGOA_IEEE/prepped/transactions.csv"
TARGET = ROOT / "data/HHGOA_IEEE/prepped/next_edges.csv"


def main() -> None:
    rows = pd.read_csv(
        SOURCE, usecols=["TransactionID", "TransactionDT", "card_key"],
        dtype={"TransactionID": str, "TransactionDT": float, "card_key": str},
    )
    rows.sort_values(["card_key", "TransactionDT", "TransactionID"], inplace=True)
    next_id = rows.groupby("card_key", sort=False)["TransactionID"].shift(-1)
    edges = pd.DataFrame({"from_txn": rows["TransactionID"], "to_txn": next_id})
    edges.dropna(inplace=True)
    edges.to_csv(TARGET, index=False)
    print(f"next_edges.csv: {len(edges):,} within-card edges")


if __name__ == "__main__":
    main()
