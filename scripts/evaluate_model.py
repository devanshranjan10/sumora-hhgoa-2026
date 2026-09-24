"""Compare historical decision models with chronological, disjoint windows.

Run: ./.venv/bin/python scripts/evaluate_model.py
Only supplied closed-case outcomes are labels. HHG cases are never training rows.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.calibrator import _validation_report
from agent.features import FEATURE_CONTRACT, feature_vector
from bench.dataset import DATA_DIR, load_dataset, norm_txn_id

OUT = ROOT / "eval" / "model_v2"


def build_rows():
    idx = load_dataset()
    rows = []
    for row in csv.DictReader((DATA_DIR / "closed_cases_history.csv").open()):
        txns = [norm_txn_id(t) for t in row["txn_ids"].split("|") if t]
        t = min(txns, key=lambda t: idx.txn_ts[t])
        rows.append({"case_id": row["case_id"], "txn_id": t,
                     "ts": idx.txn_ts[t], "closed_at": row["closed_at"],
                     "y": int(row["outcome"] == "confirmed_fraud"),
                     "x": feature_vector(idx, t)["features"]})
    return rows


def report(y, p):
    return {**_validation_report(p.tolist(), y.tolist()),
            "average_precision": float(average_precision_score(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "fraud_count": int(y.sum())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "historical_features.json"
    # Rebuild to keep source changes from silently reusing old features.
    rows = build_rows()
    cache.write_text(json.dumps(rows))
    splits = {
        "train": [r for r in rows if r["ts"] < "2016-09-01" and r["closed_at"] < "2016-09-01"],
        "calibration": [r for r in rows if "2016-09-01" <= r["ts"] < "2016-09-16" and r["closed_at"] < "2016-09-16"],
        "selection": [r for r in rows if "2016-09-16" <= r["ts"] < "2016-10-01" and r["closed_at"] < "2016-10-01"],
        "test": [r for r in rows if "2016-10-01" <= r["ts"] < "2016-11-01"],
    }
    arrays = {k: (np.asarray([r["x"] for r in v]), np.asarray([r["y"] for r in v]))
              for k, v in splits.items()}
    # Shared transaction IDs across windows would invalidate the split.
    seen = set()
    for name, rs in splits.items():
        tids = {r["txn_id"] for r in rs}
        assert not (seen & tids), f"transaction overlap in {name}"
        seen.update(tids)
    candidates = {
        "logistic": LogisticRegression(max_iter=2000, random_state=42),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=10, learning_rate=0.05, early_stopping=False, random_state=42),
    }
    selection = {}
    models = {}
    for name, model in candidates.items():
        model.fit(*arrays["train"])
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(model.predict_proba(arrays["calibration"][0])[:, 1], arrays["calibration"][1])
        p = iso.predict(model.predict_proba(arrays["selection"][0])[:, 1])
        selection[name] = report(arrays["selection"][1], p)
        models[name] = (model, iso)
    winner = min(selection, key=lambda name: selection[name]["brier"])
    model, iso = models[winner]
    # Winner is locked before accessing test labels or calculating test metrics.
    (OUT / "selection.json").write_text(json.dumps({"winner": winner, "reports": selection}, indent=2))
    X, y = arrays["test"]
    p = iso.predict(model.predict_proba(X)[:, 1])
    prior = X[:, 5]
    constant = np.full(len(y), arrays["train"][1].mean())
    rng = np.random.default_rng(42)
    auc_delta = []
    brier_delta = []
    for _ in range(1000):
        ids = rng.integers(0, len(y), len(y))
        if len(set(y[ids])) < 2:
            continue
        auc_delta.append(roc_auc_score(y[ids], p[ids]) - roc_auc_score(y[ids], prior[ids]))
        brier_delta.append(float(np.mean((p[ids] - y[ids]) ** 2 - (prior[ids] - y[ids]) ** 2)))
    summary = {
        "feature_contract": FEATURE_CONTRACT,
        "feature_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "selection_metric": "September Brier score", "winner": winner,
        "test": report(y, p), "bank_risk_baseline": report(y, prior),
        "training_prevalence_baseline": report(y, constant),
        "paired_bootstrap_95ci_vs_bank": {
            "auc_delta": np.quantile(auc_delta, [0.025, 0.975]).tolist(),
            "brier_delta": np.quantile(brier_delta, [0.025, 0.975]).tolist()},
        "limits": ["Selected closed investigations are not a representative sample of all transactions.",
                   "No labeled HHG benchmark outcomes or external competitor results are available.",
                   "Bootstrap resamples cases, so related customers may reduce effective sample size."],
    }
    joblib.dump({"model": model, "isotonic": iso, "feature_contract": FEATURE_CONTRACT,
                 "calibration_scores": model.predict_proba(arrays["calibration"][0])[:, 1],
                 "calibration_y": arrays["calibration"][1], "report": summary}, OUT / "candidate.joblib")
    (OUT / "report.json").write_text(json.dumps(summary, indent=2))
    (OUT / "predictions.json").write_text(json.dumps([
        {"case_id": r["case_id"], "y": int(yi), "p": float(pi), "bank_risk": float(bi)}
        for r, yi, pi, bi in zip(splits["test"], y, p, prior)], indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
