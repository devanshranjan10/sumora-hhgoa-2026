"""Chronologically selected decision model, with calibration uncertainty.

Load only the project-generated artifact. Joblib files can execute Python when
loaded and must never come from an untrusted upload or URL.
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from agent.calibrator import CalibratedP
from agent.features import FEATURE_CONTRACT
from agent.source_features import SourceFeatureStore
from bench.dataset import DATA_DIR

MODEL_PATH = Path(__file__).resolve().parents[1] / "eval/overlap_model/candidate.joblib"


class DecisionModel:
    def __init__(self, artifact: Path = MODEL_PATH, data_dir: Path = DATA_DIR):
        state = joblib.load(artifact)
        if state["feature_contract"] != FEATURE_CONTRACT:
            raise ValueError("Decision model and runtime feature contracts differ")
        self.model = state["model"]
        self.feature_indices = state.get("feature_indices")
        self.calibrator = state["calibrator"]
        self.calibration_method = state["calibration_method"]
        self.metadata = state["report"]
        self.rich = state.get("rich")
        self.data_dir = data_dir
        self.source_store = None
        scores, y = state["calibration_scores"], state["calibration_y"]
        self.bootstrap = self._bootstrap(scores, y, self.calibration_method)
        self.rich_bootstrap = (
            self._bootstrap(self.rich["calibration_scores"], self.rich["calibration_y"],
                            self.rich["calibration_method"])
            if self.rich else []
        )

    @staticmethod
    def _bootstrap(scores, y, method: str) -> list[IsotonicRegression | LogisticRegression]:
        rng = np.random.default_rng(42)
        fits = []
        for ids in (rng.integers(0, len(y), len(y)) for _ in range(200)):
            sample_scores, sample_y = scores[ids], y[ids]
            counts = np.bincount(sample_y, minlength=2)
            if min(counts) == 0:
                continue
            weights = np.where(sample_y == 1, .5 / counts[1], .5 / counts[0]) * len(sample_y)
            if method == "isotonic":
                fit = IsotonicRegression(out_of_bounds="clip").fit(
                    sample_scores, sample_y, sample_weight=weights)
            else:
                fit = LogisticRegression(C=10, random_state=42).fit(
                    sample_scores.reshape(-1, 1), sample_y, sample_weight=weights)
            fits.append(fit)
        return fits

    @staticmethod
    def _calibrate(fit, method: str, score):
        return (fit.predict(score) if method == "isotonic" else
                fit.predict_proba(score.reshape(-1, 1))[:, 1])

    @staticmethod
    def _predict(model, calibrator, method, bootstrap, x: list[float]) -> CalibratedP:
        score = model.predict_proba([x])[:, 1]
        p = float(np.clip(DecisionModel._calibrate(calibrator, method, score)[0], 0.05, 0.95))
        predictions = [float(np.clip(DecisionModel._calibrate(fit, method, score)[0], 0.05, 0.95))
                       for fit in bootstrap]
        lo, hi = np.quantile(predictions, [0.05, 0.95])
        return CalibratedP(p, min(float(lo), p), max(float(hi), p))

    def predict_ci(self, x: list[float], txn_id: str | None = None) -> CalibratedP:
        if self.feature_indices is not None:
            x = [x[i] for i in self.feature_indices]
        if self.rich and txn_id:
            if self.source_store is None:
                self.source_store = SourceFeatureStore(self.data_dir)
            row = self.source_store.get(txn_id)
            if row is not None:
                numeric = [
                    float(row[key]) if row.get(key) not in ("", None) else np.nan
                    for key in self.rich["numeric_names"]
                ]
                categorical = [row.get(key, "") or ""
                               for key in self.rich["categorical_names"]]
                encoded = self.rich["encoder"].transform([categorical])[0]
                rich_x = np.asarray([*x, *numeric, *encoded], dtype=np.float32).tolist()
                return self._predict(self.rich["model"], self.rich["calibrator"],
                                     self.rich["calibration_method"], self.rich_bootstrap, rich_x)
        return self._predict(self.model, self.calibrator, self.calibration_method,
                             self.bootstrap, x)

    def report(self) -> dict:
        return self.rich["report"] if self.rich else self.metadata["test"]
