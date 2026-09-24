"""Read one supplied transaction's raw fields without loading the full CSV."""

from __future__ import annotations

import csv
from pathlib import Path

from bench.dataset import DATA_DIR


class SourceFeatureStore:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.transactions = data_dir / "transactions.csv"
        self.identity = data_dir / "identity.csv"
        self.txn_header, self.txn_offsets = self._index(self.transactions)
        if "card_key" in self.txn_header:
            raise ValueError("Decision model requires raw transactions.csv, not prepped data")
        self.identity_header, self.identity_offsets = self._index(self.identity)

    @staticmethod
    def _index(path: Path) -> tuple[list[str], dict[str, int]]:
        offsets = {}
        with path.open("rb") as source:
            header = next(csv.reader([source.readline().decode("utf-8-sig")]))
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                txn_id = line.split(b",", 1)[0].decode("ascii")
                if not txn_id.isdecimal():
                    raise ValueError(f"Unsupported multiline CSV row in {path} at {offset}")
                offsets[txn_id] = offset
        return header, offsets

    @staticmethod
    def _row(path: Path, header: list[str], offset: int) -> dict[str, str]:
        with path.open("rb") as source:
            source.seek(offset)
            values = next(csv.reader([source.readline().decode("utf-8")]))
        if len(values) != len(header):
            raise ValueError(f"Malformed source CSV row in {path} at {offset}")
        return dict(zip(header, values))

    def get(self, txn_id: str) -> dict[str, str] | None:
        raw = txn_id.removeprefix("T").lstrip("0")
        offset = self.txn_offsets.get(raw)
        if offset is None:
            return None
        row = self._row(self.transactions, self.txn_header, offset)
        identity_offset = self.identity_offsets.get(raw)
        if identity_offset is not None:
            row.update(self._row(self.identity, self.identity_header, identity_offset))
        return row
