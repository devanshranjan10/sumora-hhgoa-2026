"""Preprocess HHGOA_IEEE CSVs for TigerGraph loading.

The TigerGraph loading grammar (4.2) does not support string concatenation or
IIF() in LOAD expressions, so every derived column is materialized here and the
loading jobs reference plain $"col" tokens only.

Outputs (data/HHGOA_IEEE/prepped/):
  transactions.csv  + card_key, email_p, email_r, cluster_key
  identity.csv      + dp_key          (empty where no usable device)
  case_pack.csv     + prior_score, trigger_tag

Card keys use supplied closed/case links plus exact card1..card6 fingerprints.
Unmatched fingerprints receive internal U-ids; we never guess K1/K2.

Run:  ./.venv/bin/python scripts/prep_load_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.card_identity import CARD_FIELDS, learn_card_fingerprints, resolve_card

SRC = ROOT / "data" / "HHGOA_IEEE"
OUT = SRC / "prepped"


def trim(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def fill_cols(chunk: pd.DataFrame, numeric: set[str]) -> pd.DataFrame:
    """NaN -> "0" for numeric columns, "" for everything else (dtype=str in)."""
    for c in chunk.columns:
        chunk[c] = chunk[c].fillna("0" if c in numeric else "")
    return chunk


NUMERIC_TXN = {
    "TransactionDT", "TransactionAmt", "card1", "card2", "card3", "card5",
    "risk_score", "dist1", "dist2",
    *(f"C{i}" for i in range(1, 15)),
    *(f"D{i}" for i in range(1, 16)),
    "V12", "V29", "V45", "V53", "V57", "V62", "V65", "V70", "V75", "V94",
    "V95", "V96", "V97", "V101", "V102", "V107", "V127", "V128", "V130",
    "V131", "V137", "V187", "V200", "V201", "V258", "V279", "V294", "V307",
    "V308", "V310",
}


def prep_transactions() -> None:
    cols = None
    out_parts: list[pd.DataFrame] = []
    card_mapping, authoritative_links = learn_card_fingerprints(SRC)
    for chunk in pd.read_csv(SRC / "transactions.csv", chunksize=200_000, dtype=str):
        if cols is None:
            cols = list(chunk.columns)
        identity_columns = ("TransactionID", "customer_id") + CARD_FIELDS
        identity_values = (trim(chunk[column]) for column in identity_columns)
        card_keys = [
            resolve_card(dict(zip(identity_columns, values)), card_mapping, authoritative_links)
            for values in zip(*identity_values)
        ]
        fill_cols(chunk, NUMERIC_TXN)
        chunk["card_key"] = card_keys
        ep = trim(chunk["P_emaildomain"])
        er = trim(chunk["R_emaildomain"])
        chunk["email_p"] = ("p:" + ep).where(ep != "", "")
        chunk["email_r"] = ("r:" + er).where(er != "", "")
        chunk["p_domain"] = ep          # trimmed domain (vertex id safe)
        chunk["r_domain"] = er
        a1 = trim(chunk["addr1"])
        chunk["addr1_s"] = a1           # trimmed addr1 (BillingRegion id)
        ok = (a1 != "") & (ep != "")
        chunk["cluster_key"] = (
            "IC:" + trim(chunk["card1"]) + "|" + a1 + "|" + ep
        ).where(ok, "")
        out_parts.append(chunk)
    OUT.mkdir(exist_ok=True)
    pd.concat(out_parts, ignore_index=True).to_csv(
        OUT / "transactions.csv", index=False
    )
    print(f"transactions.csv: {sum(len(p) for p in out_parts):,} rows -> prepped/")


def prep_identity() -> None:
    parts = []
    for chunk in pd.read_csv(SRC / "identity.csv", chunksize=200_000, dtype=str):
        fill_cols(chunk, set())
        d = trim(chunk["DeviceInfo"])
        ok = (d != "") & (d != "NotFound")
        chunk["dp_key"] = (
            "DP:" + d + "|" + trim(chunk["id_31"]) + "|" + trim(chunk["id_33"])
        ).where(ok, "")
        parts.append(chunk)
    OUT.mkdir(exist_ok=True)
    pd.concat(parts, ignore_index=True).to_csv(OUT / "identity.csv", index=False)
    print(f"identity.csv: {sum(len(p) for p in parts):,} rows -> prepped/")


def prep_case_pack() -> None:
    df = pd.read_csv(SRC / "case_pack.csv", dtype=str)
    fill_cols(df, {"risk_score"})
    rs = trim(df["risk_score"])
    df["prior_score"] = pd.to_numeric(rs.where(rs != "", "0.0"), errors="coerce").fillna(0.0)
    df["trigger_tag"] = "trigger:" + trim(df["trigger_type"])
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "case_pack.csv", index=False)
    print(f"case_pack.csv: {len(df):,} rows -> prepped/")


def prep_closed_cases() -> None:
    # Pipe-splitting happens in the loader via flatten(); numeric empties are
    # filled so to_int/to_float never see "".
    df = pd.read_csv(SRC / "closed_cases_history.csv", dtype=str)
    fill_cols(df, {"n_txns", "exposure_usd"})
    df.to_csv(OUT / "closed_cases_history.csv", index=False)
    print(f"closed_cases_history.csv: {len(df):,} rows -> prepped/")


def main() -> int:
    prep_transactions()
    from scripts.prep_next_edges import main as prep_next_edges
    prep_next_edges()
    prep_identity()
    prep_case_pack()
    prep_closed_cases()
    return 0


if __name__ == "__main__":
    sys.exit(main())
