"""Evaluate source-transaction features with a balanced temporal holdout.

Only the supplied closed-case outcomes are labels. The case pack is prediction
data, and the October outcomes remain untouched until model selection ends.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.features import feature_vector
from bench.dataset import DATA_DIR, load_dataset
from bench.run import load_cases


HIST = ROOT / "eval/model_v2/historical_features.json"
V_COLS = [
    12, 29, 45, 53, 57, 62, 65, 70, 75, 94, 95, 96, 97, 101, 102,
    107, 127, 128, 130, 131, 137, 187, 200, 201, 258, 279, 294,
    307, 308, 310,
]
NUMERIC = (
    ["TransactionAmt", "dist1", "dist2"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + [f"V{i}" for i in V_COLS]
    + [f"id_{i:02d}" for i in (1, 2, 5, 6, 11)]
)
CATEGORICAL = ["ProductCD", "card4", "card6"] + [f"M{i}" for i in range(1, 10)] + ["id_15", "id_23", "DeviceType"]


def balanced_weights(y: np.ndarray) -> np.ndarray:
    return np.where(y == 1, 0.5 / (y == 1).sum(), 0.5 / (y == 0).sum()) * len(y)


def source_rows(ids: set[str]) -> dict[str, dict[str, str]]:
    rows = {}
    with (DATA_DIR / "transactions.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            tid = row["TransactionID"]
            if tid in ids:
                rows[tid] = {k: row.get(k, "") for k in NUMERIC + CATEGORICAL}
    with (DATA_DIR / "identity.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            tid = row["TransactionID"]
            if tid in rows:
                rows[tid].update({k: row.get(k, "") for k in NUMERIC + CATEGORICAL if k.startswith("id_") or k == "DeviceType"})
    missing = ids - rows.keys()
    if missing:
        raise ValueError(f"missing transaction rows: {sorted(missing)[:5]}")
    return rows


def main() -> None:
    historical = json.loads(HIST.read_text())
    cases = load_cases(DATA_DIR / "case_pack.csv")
    ids = {r["txn_id"].removeprefix("T").lstrip("0") for r in historical}
    ids |= {c.flagged_txn_id.removeprefix("T").lstrip("0") for c in cases}
    source = source_rows(ids)
    idx = load_dataset(DATA_DIR)
    exam = [
        {"case_id": c.case_id, "txn_id": c.flagged_txn_id.removeprefix("T").lstrip("0"),
         "x": feature_vector(idx, c.flagged_txn_id, c.risk_score)["features"]}
        for c in cases
    ]
    splits = {
        "train": [r for r in historical if r["ts"] < "2016-09-01" and r["closed_at"] < "2016-09-01"],
        "cal": [r for r in historical if "2016-09-01" <= r["ts"] < "2016-09-16" and r["closed_at"] < "2016-09-16"],
        "select": [r for r in historical if "2016-09-16" <= r["ts"] < "2016-10-01" and r["closed_at"] < "2016-10-01"],
        "test": [r for r in historical if "2016-10-01" <= r["ts"] < "2016-11-01"],
        "exam": exam,
    }
    def categorical(rs):
        return [[source[r["txn_id"].removeprefix("T").lstrip("0")].get(k, "") or ""
                 for k in CATEGORICAL] for r in rs]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10)
    encoder.fit(categorical(splits["train"]))

    def matrix(rs, rich: bool) -> np.ndarray:
        base = np.asarray([r["x"] for r in rs], dtype=np.float32)
        if not rich:
            return base
        nums = np.asarray([
            [float(value) if value not in ("", None) else np.nan
             for k in NUMERIC
             for value in [source[r["txn_id"].removeprefix("T").lstrip("0")].get(k, "")]]
            for r in rs
        ], dtype=np.float32)
        cats = encoder.transform(categorical(rs)).astype(np.float32)
        return np.column_stack((base, nums, cats))

    candidates = {
        "base_balanced": (False, "balanced"),
        "rich_balanced": (True, "balanced"),
        "rich_unweighted": (True, None),
    }
    selected = {}
    fitted = {}
    for name, (rich, class_weight) in candidates.items():
        X = {part: matrix(rs, rich) for part, rs in splits.items()}
        y = {part: np.asarray([r["y"] for r in rs], dtype=int)
             for part, rs in splits.items() if part != "exam"}
        model = HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=10, learning_rate=0.05,
            class_weight=class_weight, early_stopping=False, random_state=42,
        ).fit(X["train"], y["train"])
        cal_score = model.predict_proba(X["cal"])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            cal_score, y["cal"], sample_weight=balanced_weights(y["cal"])
        )
        score = model.predict_proba(X["select"])[:, 1]
        p = iso.predict(score)
        selected[name] = {
            "auc": float(roc_auc_score(y["select"], score)),
            "balanced_brier": float(brier_score_loss(y["select"], p, sample_weight=balanced_weights(y["select"]))),
        }
        fitted[name] = (model, iso, X, y)
        print(name, "selection", selected[name], flush=True)
    winner = min(selected, key=lambda n: selected[n]["balanced_brier"])
    model, iso, X, y = fitted[winner]
    test_score = model.predict_proba(X["test"])[:, 1]
    test_p = iso.predict(test_score)
    print("winner", winner, flush=True)
    print("October", {
        "auc": float(roc_auc_score(y["test"], test_score)),
        "balanced_brier": float(brier_score_loss(y["test"], test_p, sample_weight=balanced_weights(y["test"]))),
    }, flush=True)
    for threshold in (0.9, 0.95, 0.99, 0.995, 0.998):
        chosen = test_score >= threshold
        print("October tail", threshold, "count", int(chosen.sum()),
              "cleared", int(((y["test"] == 0) & chosen).sum()), flush=True)
    exam_score = model.predict_proba(X["exam"])[:, 1]
    exam_p = iso.predict(exam_score)
    for case, score, p in sorted(zip(exam, exam_score, exam_p), key=lambda x: -x[2]):
        print(case["case_id"], round(float(score), 4), round(float(p), 4), flush=True)


if __name__ == "__main__":
    main()
