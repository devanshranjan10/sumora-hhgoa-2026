"""Dataset index for the HHGOA_IEEE data (ID existence checks + stub features).

Loaded once per benchmark run. transactions.csv is 675MB / 590k rows; we scan
it once and keep only the columns the stub runner and validators need.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from bench.card_identity import learn_card_fingerprints, resolve_card

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "HHGOA_IEEE"

# Answer files refer to transactions as T0412877; transactions.csv stores 412877.
TXN_RE = re.compile(r"^T?0*(\d+)$")
CARD_RE = re.compile(r"^C\d{5}-K\d$")
CUSTOMER_RE = re.compile(r"^C\d{5}$")
CASE_ID_RE = re.compile(r"^HHG-\d{3}$")
CLOSED_CASE_RE = re.compile(r"^CC-\d{4}$")

_DATASET_START = "2016-06-01"  # README: TransactionDT is seconds from dataset start


def norm_txn_id(raw) -> str:
    """Normalize a transaction reference to the zero-padded T-form."""
    m = TXN_RE.match(str(raw).strip())
    if not m:
        return str(raw).strip()
    return f"T{int(m.group(1)):07d}"


def txn_id_variants(tid: str) -> Tuple[str, str]:
    """(T-form, raw csv form) for one transaction id."""
    m = TXN_RE.match(tid)
    if not m:
        return tid, tid
    n = int(m.group(1))
    return f"T{n:07d}", str(n)


def dt_to_date(seconds: float) -> str:
    """TransactionDT (seconds from dataset start) -> YYYY-MM-DD."""
    import datetime as _dt

    base = _dt.datetime.fromisoformat(_DATASET_START)
    return (base + _dt.timedelta(seconds=seconds)).date().isoformat()


@dataclass
class DatasetIndex:
    txn_ids: Set[str] = field(default_factory=set)          # T-form
    txn_amount: Dict[str, float] = field(default_factory=dict)
    txn_dt: Dict[str, float] = field(default_factory=dict)
    txn_card: Dict[str, str] = field(default_factory=dict)
    txn_online: Dict[str, bool] = field(default_factory=dict)
    txn_ts: Dict[str, str] = field(default_factory=dict)
    txn_risk: Dict[str, float] = field(default_factory=dict)
    txn_customer: Dict[str, str] = field(default_factory=dict)
    customer_txns: Dict[str, List[str]] = field(default_factory=dict)
    card_customer: Dict[str, str] = field(default_factory=dict)
    customer_cards: Dict[str, List[str]] = field(default_factory=dict)
    card_txns: Dict[str, List[str]] = field(default_factory=dict)
    txn_device: Dict[str, str] = field(default_factory=dict)
    device_profiles: Set[str] = field(default_factory=set)
    closed_case_ids: Set[str] = field(default_factory=set)
    closed_case_cards: Dict[str, Set[str]] = field(default_factory=dict)
    closed_case_txns: Dict[str, Set[str]] = field(default_factory=dict)
    # outcome semantics: closed-case ids whose bank-confirmed outcome is fraud
    # (cleared = complement); populated by load_dataset from closed_cases_history
    closed_case_outcome_fraud: Set[str] = field(default_factory=set)
    closed_case_closed_at: Dict[str, str] = field(default_factory=dict)

    def txn_exists(self, tid: str) -> bool:
        return norm_txn_id(tid) in self.txn_ids

    def card_exists(self, cid: str) -> bool:
        return cid in self.card_customer

    def customer_exists(self, cid: str) -> bool:
        return cid in self.customer_cards

    def closed_case_exists(self, ccid: str) -> bool:
        return ccid in self.closed_case_ids

    def device_profile_exists(self, profile: str) -> bool:
        return profile in self.device_profiles

    def amount(self, tid: str) -> float:
        return self.txn_amount.get(norm_txn_id(tid), 0.0)

    def date_of(self, tid: str) -> str:
        if norm_txn_id(tid) in self.txn_ts:
            return self.txn_ts[norm_txn_id(tid)][:10]
        return dt_to_date(self.txn_dt.get(norm_txn_id(tid), 0.0))

    def exposure(self, tids: List[str]) -> float:
        return round(sum(abs(self.amount(t)) for t in tids), 2)


def _device_profile(row: Dict[str, str]) -> str:
    """DeviceProfile = DeviceInfo + OS + browser + screen (README graph schema)."""
    parts = [
        row.get("DeviceInfo") or "",
        row.get("id_30") or "",
        row.get("id_31") or "",
        row.get("id_33") or "",
    ]
    if not any(parts):
        return ""
    return " | ".join(p for p in parts if p)


def load_dataset(data_dir: Optional[Path] = None) -> DatasetIndex:
    data_dir = data_dir or DATA_DIR
    idx = DatasetIndex()
    card_mapping, authoritative_links = learn_card_fingerprints(data_dir)

    # identity.csv: TransactionID -> device profile
    identity_device: Dict[str, str] = {}
    with (data_dir / "identity.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            prof = _device_profile(row)
            if prof:
                identity_device[row["TransactionID"]] = prof
                idx.device_profiles.add(prof)

    # transactions.csv: one pass, keep only what we need
    with (data_dir / "transactions.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            raw = row["TransactionID"]
            t = norm_txn_id(raw)
            idx.txn_ids.add(t)
            try:
                idx.txn_amount[t] = float(row["TransactionAmt"])
            except (TypeError, ValueError):
                idx.txn_amount[t] = 0.0
            try:
                idx.txn_dt[t] = float(row["TransactionDT"])
            except (TypeError, ValueError):
                idx.txn_dt[t] = 0.0
            customer = row["customer_id"]
            idx.txn_customer[t] = customer
            idx.txn_card[t] = resolve_card(row, card_mapping, authoritative_links)
            idx.customer_txns.setdefault(customer, []).append(t)
            idx.customer_cards.setdefault(customer, [])
            idx.txn_ts[t] = row["ts"]
            idx.txn_risk[t] = float(row["risk_score"])
            idx.txn_online[t] = row["channel"] == "online"
            if raw in identity_device:
                idx.txn_device[t] = identity_device[raw]

    # closed_cases_history.csv: authoritative card/customer ids + closed case ids
    with (data_dir / "closed_cases_history.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            ccid = row["case_id"]
            idx.closed_case_ids.add(ccid)
            idx.closed_case_closed_at[ccid] = row["closed_at"]
            if row["outcome"] == "confirmed_fraud":
                idx.closed_case_outcome_fraud.add(ccid)
            card, cust = row["card_id"], row["customer_id"]
            idx.card_customer.setdefault(card, cust)
            idx.customer_cards.setdefault(cust, [])
            if card not in idx.customer_cards[cust]:
                idx.customer_cards[cust].append(card)
            txns = {norm_txn_id(t) for t in row["txn_ids"].split("|") if t}
            idx.closed_case_txns[ccid] = txns
            linked = {card}
            for t in txns:
                if t in idx.txn_ids:
                    idx.txn_card[t] = card
            idx.closed_case_cards[ccid] = linked

    # case_pack.csv: the 20 benchmark cases' card/customer ids are dataset ids too
    case_pack = data_dir / "case_pack.csv"
    if case_pack.exists():
        with case_pack.open(newline="") as fh:
            for row in csv.DictReader(fh):
                card, cust = row["card_id"], row["customer_id"]
                idx.txn_card[norm_txn_id(row["flagged_txn_id"])] = card
                idx.card_customer.setdefault(card, cust)
                idx.customer_cards.setdefault(cust, [])
                if card not in idx.customer_cards[cust]:
                    idx.customer_cards[cust].append(card)

    # card -> txns map (only for cards we know; bounded memory)
    known_cards = set(idx.card_customer)
    for t, c in idx.txn_card.items():
        if c in known_cards:
            idx.card_txns.setdefault(c, []).append(t)

    return idx
