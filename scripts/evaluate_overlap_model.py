"""Test fraud ranking where historical fraud and cleared cases overlap.

The supplied closed-case cohort has no cleared cases below bank risk 0.80.
This experiment excludes that score from features and fits on the overlap
region only. October remains a held-out evaluation; HHG outcomes are unknown.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.dataset import DATA_DIR, load_dataset
from bench.run import load_cases
from agent.features import FEATURE_CONTRACT, feature_vector
from scripts.evaluate_rich_model import CATEGORICAL, HIST, NUMERIC, balanced_weights, source_rows
from scripts.evaluate_model import build_rows

OUT = ROOT / "eval" / "overlap_model"


def main() -> None:
    if HIST.exists():
        historical = json.loads(HIST.read_text())
    else:
        historical = build_rows()
        HIST.parent.mkdir(parents=True, exist_ok=True)
        HIST.write_text(json.dumps(historical))
    cases = load_cases(DATA_DIR / "case_pack.csv")
    identifiers = {r["txn_id"].removeprefix("T").lstrip("0") for r in historical}
    identifiers |= {c.flagged_txn_id.removeprefix("T").lstrip("0") for c in cases}
    source = source_rows(identifiers)
    index = load_dataset(DATA_DIR)
    exam = [
        {"case_id": c.case_id, "txn_id": c.flagged_txn_id.removeprefix("T").lstrip("0"),
         "x": feature_vector(index, c.flagged_txn_id, c.risk_score)["features"]}
        for c in cases
    ]
    splits = {
        "train": [r for r in historical if r["ts"] < "2016-09-01"
                  and r["closed_at"] < "2016-09-01" and r["x"][5] >= .8],
        "cal": [r for r in historical if "2016-09-01" <= r["ts"] < "2016-09-16"
                and r["closed_at"] < "2016-09-16" and r["x"][5] >= .8],
        "select": [r for r in historical if "2016-09-16" <= r["ts"] < "2016-10-01"
                   and r["closed_at"] < "2016-10-01" and r["x"][5] >= .8],
        "test": [r for r in historical if "2016-10-01" <= r["ts"] < "2016-11-01"
                 and r["x"][5] >= .8],
        "exam": exam,
    }

    def categorical(rows):
        return [[source[r["txn_id"].removeprefix("T").lstrip("0")].get(k, "") or ""
                 for k in CATEGORICAL] for r in rows]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10)
    encoder.fit(categorical(splits["train"]))

    def matrix(rows, rich):
        base = np.asarray([[v for i, v in enumerate(r["x"]) if i != 5]
                           for r in rows], dtype=np.float32)
        if not rich:
            return base
        numeric = np.asarray([
            [float(value) if value not in ("", None) else np.nan
             for key in NUMERIC
             for value in [source[r["txn_id"].removeprefix("T").lstrip("0")].get(key, "")]]
            for r in rows
        ], dtype=np.float32)
        return np.column_stack((base, numeric,
                                encoder.transform(categorical(rows)).astype(np.float32)))

    targets = {part: np.asarray([r["y"] for r in rows], dtype=int)
               for part, rows in splits.items() if part != "exam"}
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    candidate = {}
    for name, rich, model in [
        ("base_hgb", False, HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=10, learning_rate=.05, class_weight="balanced",
            early_stopping=False, random_state=42)),
        ("rich_hgb", True, HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, min_samples_leaf=30,
            l2_regularization=10, learning_rate=.05, class_weight="balanced",
            early_stopping=False, random_state=42)),
    ]:
        arrays = {part: matrix(rows, rich) for part, rows in splits.items()}
        model.fit(arrays["train"], targets["train"])
        cal_scores = model.predict_proba(arrays["cal"])[:, 1]
        calibrators = {
            "isotonic": IsotonicRegression(out_of_bounds="clip").fit(
                cal_scores, targets["cal"],
                sample_weight=balanced_weights(targets["cal"])),
            "sigmoid": LogisticRegression(C=10, random_state=42).fit(
                cal_scores.reshape(-1, 1), targets["cal"],
                sample_weight=balanced_weights(targets["cal"])),
        }
        select_scores = model.predict_proba(arrays["select"])[:, 1]
        def calibrated(kind, scores):
            fit = calibrators[kind]
            return (fit.predict(scores) if kind == "isotonic" else
                    fit.predict_proba(scores.reshape(-1, 1))[:, 1])

        selection_brier = {
            kind: float(brier_score_loss(
                targets["select"], np.clip(calibrated(kind, select_scores), .05, .95),
                sample_weight=balanced_weights(targets["select"])))
            for kind in calibrators
        }
        calibration_kind = min(selection_brier, key=selection_brier.get)
        calibrator = calibrators[calibration_kind]
        result = {"split_counts": {part: len(rows) for part, rows in splits.items()},
                  "feature_count": int(arrays["train"].shape[1]),
                  "calibration_method": calibration_kind,
                  "selection_brier_by_calibration": selection_brier}
        for part in ("select", "test"):
            score = model.predict_proba(arrays[part])[:, 1]
            p = np.clip(calibrated(calibration_kind, score), 0.05, 0.95)
            y = targets[part]
            result[part] = {"auc": float(roc_auc_score(y, score)),
                            "balanced_brier": float(brier_score_loss(
                                y, p, sample_weight=balanced_weights(y))),
                            "prevalence": float(y.mean())}
        exam_scores = model.predict_proba(arrays["exam"])[:, 1]
        exam_p = np.clip(calibrated(calibration_kind, exam_scores), 0.05, 0.95)
        result["exam"] = [
            {"case_id": r["case_id"], "score": float(s), "p": float(p)}
            for r, s, p in zip(exam, exam_scores, exam_p)
        ]
        results[name] = result
        if name == "base_hgb":
            candidate = {
                "feature_contract": FEATURE_CONTRACT,
                "feature_indices": [0, 1, 2, 3, 4, 6],
                "model": model,
                "calibrator": calibrator,
                "calibration_method": calibration_kind,
                "calibration_scores": cal_scores,
                "calibration_y": targets["cal"],
                "report": {"test": result["test"]},
            }
        else:
            candidate["rich"] = {
                "model": model,
                "calibrator": calibrator,
                "calibration_method": calibration_kind,
                "encoder": encoder,
                "numeric_names": NUMERIC,
                "categorical_names": CATEGORICAL,
                "calibration_scores": cal_scores,
                "calibration_y": targets["cal"],
                "report": result["test"],
            }
        print(name, result["select"], result["test"], flush=True)
        print([(r["case_id"], round(r["p"], 3))
               for r in sorted(result["exam"], key=lambda r: r["p"])], flush=True)
    (OUT / "report.json").write_text(json.dumps(results, indent=2))
    joblib.dump(candidate, OUT / "candidate.joblib")


if __name__ == "__main__":
    main()
