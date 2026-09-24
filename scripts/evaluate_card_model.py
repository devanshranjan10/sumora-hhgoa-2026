"""Evaluate time-aware card behavior on the historical label-overlap cohort."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.card_features import CARD_FEATURE_NAMES, CardHistory
from agent.features import feature_vector
from bench.dataset import DATA_DIR, load_dataset
from bench.run import load_cases
from scripts.evaluate_rich_model import HIST, balanced_weights

OUT = ROOT / "eval" / "card_model"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-cache", type=Path)
    args = parser.parse_args()
    idx = joblib.load(args.index_cache) if args.index_cache else load_dataset(DATA_DIR)
    history = CardHistory(idx)
    rows = json.loads(HIST.read_text())
    exam = [
        {"case_id": case.case_id, "txn_id": case.flagged_txn_id,
         "x": feature_vector(idx, case.flagged_txn_id, case.risk_score)["features"]}
        for case in load_cases(DATA_DIR / "case_pack.csv")
    ]
    splits = {
        "train": [r for r in rows if r["ts"] < "2016-09-01"
                  and r["closed_at"] < "2016-09-01" and r["x"][5] >= .8],
        "cal": [r for r in rows if "2016-09-01" <= r["ts"] < "2016-09-16"
                and r["closed_at"] < "2016-09-16" and r["x"][5] >= .8],
        "select": [r for r in rows if "2016-09-16" <= r["ts"] < "2016-10-01"
                   and r["closed_at"] < "2016-10-01" and r["x"][5] >= .8],
        "test": [r for r in rows if "2016-10-01" <= r["ts"] < "2016-11-01"
                 and r["x"][5] >= .8],
        "exam": exam,
    }
    features = {}
    for name, part in splits.items():
        features[name] = {
            "base": np.asarray([[v for i, v in enumerate(r["x"]) if i != 5]
                                for r in part], dtype=np.float32),
            "card": np.asarray([history.features(r["txn_id"]) for r in part], dtype=np.float32),
        }
    labels = {part: np.asarray([r["y"] for r in splits[part]], dtype=int)
              for part in ("train", "cal", "select", "test")}
    reports = {}
    for name in ("base", "card", "combined"):
        def matrix(part):
            if name == "combined":
                return np.column_stack((features[part]["base"], features[part]["card"]))
            return features[part][name]

        model = HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=10, learning_rate=.05, class_weight="balanced",
            early_stopping=False, random_state=42,
        ).fit(matrix("train"), labels["train"])
        cal_scores = model.predict_proba(matrix("cal"))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            cal_scores, labels["cal"], sample_weight=balanced_weights(labels["cal"])
        )
        report = {"feature_names": list(CARD_FEATURE_NAMES) if name == "card" else name,
                  "split_counts": {part: len(rs) for part, rs in splits.items()}}
        for part in ("select", "test"):
            raw = model.predict_proba(matrix(part))[:, 1]
            p = iso.predict(raw)
            y = labels[part]
            report[part] = {"auc": float(roc_auc_score(y, raw)),
                            "balanced_brier": float(brier_score_loss(
                                y, p, sample_weight=balanced_weights(y)))}
        raw = model.predict_proba(matrix("exam"))[:, 1]
        p = iso.predict(raw)
        report["exam"] = [
            {"case_id": row["case_id"], "raw": float(s), "p": float(q)}
            for row, s, q in zip(splits["exam"], raw, p)
        ]
        reports[name] = report
        print(name, report["select"], report["test"], flush=True)
        print([(r["case_id"], round(r["p"], 3))
               for r in sorted(report["exam"], key=lambda r: r["p"])], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
