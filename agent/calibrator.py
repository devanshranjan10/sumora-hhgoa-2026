"""Shared probability and stance functions, plus the archived logistic model.

The live scorer is ``agent.decision_model.DecisionModel``. The
``IsotonicCalibrator`` class remains for the earlier baseline comparison.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# Stance likelihoods (documented in agent/policy.yaml; consumed by
# update_with_stance). NOT learned from history: in closed_cases_history the
# recorded stance is outcome-derivative (fraud cases were resolved via denial,
# cleared via confirmation), so learning it would be label leakage.
STANCE_LIKELIHOODS = {
    "denied":    {"tpr": 0.90, "fpr": 0.05},   # P(implicating | fraud/legit)
    "confirmed": {"tpr": 0.05, "fpr": 0.90},
    "silent":    {"tpr": 0.30, "fpr": 0.30},   # weak, nearly uninformative
}


def memory_signal(n_fraud: int, n_cleared: int) -> float:
    """Signed, volume-normalized memory contrast.

    (F - C) / sqrt(F + C + 1): direction carries the signal, volume is
    normalized away. Raw counts co-vary inside dense identity clusters
    (both rise with cluster activity); the contrast isolates what memory
    actually knows: which outcome dominates this device's history.
    """
    import math as _m
    return (n_fraud - n_cleared) / _m.sqrt(n_fraud + n_cleared + 1.0)


def case_features(small_auth_rate: float, amount_z: float, online: bool,
                  device_novelty: int, night_txn: int, prior_score: float,
                  mem_signal_val: float) -> List[float]:
    """Discriminative feature vector (order = contract, do not reorder).

    Design note: the v1 vector (log amount-ratio + raw counts) collapsed to a
    base-rate constant - every case scored ~0.93 because the features had no
    spread across cases. These features are chosen for cross-case variance:
      small_auth_rate - card-testing signature (R5): many <=$5 auths
      amount_z        - (amount - card_mean)/card_std, outlier power of THIS txn
      device_novelty  - device never seen on this card (R2)
      night_txn       - authorized 00:00-05:00 local
      mem_signal      - signed memory contrast (see memory_signal)
    Customer stance is deliberately NOT a learned feature - STANCE_LIKELIHOODS.
    """
    return [
        float(min(small_auth_rate, 1.0)),
        float(np.clip(amount_z, -5.0, 15.0)),
        1.0 if online else 0.0,
        float(device_novelty),
        float(night_txn),
        float(prior_score),
        float(np.clip(mem_signal_val, -3.0, 3.0)),
    ]


FEATURE_NAMES = [
    "small_auth_rate", "amount_z", "online", "device_novelty", "night_txn",
    "prior_score", "mem_signal",
]


def update_with_stance(p_base: float, stance: str) -> float:
    """Bayes update of the calibrated base p with the customer-stance signal.

    Keeps p a single quantity: the calibrator owns both the learned isotonic
    map and this documented likelihood-ratio update; the LLM never emits p.
    """
    like = STANCE_LIKELIHOODS.get(stance, STANCE_LIKELIHOODS["silent"])
    num = p_base * like["tpr"]
    den = num + (1.0 - p_base) * like["fpr"]
    return float(np.clip(num / den if den > 0 else p_base, 0.0, 1.0))


@dataclass
class CalibratedP:
    p: float
    lo: float  # 90% CI lower
    hi: float  # 90% CI upper


def _new_iso() -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)


class IsotonicCalibrator:
    """Logistic scorer + isotonic calibration, with bootstrap CIs."""

    def __init__(self) -> None:
        self._lr: Optional[LogisticRegression] = None
        self._iso = _new_iso()
        self._boot: List[IsotonicRegression] = []
        self._report: Dict[str, Any] = {}

    # -- internal ----------------------------------------------------------
    def _score(self, X: List[List[float]]) -> np.ndarray:
        assert self._lr is not None
        return self._lr.decision_function(np.asarray(X, dtype=float))

    # -- training ----------------------------------------------------------
    def fit(self, X: List[List[float]], y: List[int],
            X_val: Optional[List[List[float]]] = None,
            y_val: Optional[List[int]] = None,
            n_boot: int = 200, seed: int = 42) -> Dict[str, Any]:
        Xa = np.asarray(X, dtype=float)
        ya = np.asarray(y, dtype=int)
        self._lr = LogisticRegression(max_iter=1000, random_state=seed)
        self._lr.fit(Xa, ya)
        s = self._score(X)
        self._iso.fit(s, ya)
        rng = np.random.default_rng(seed)
        self._boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(ya), len(ya))
            iso = _new_iso()
            iso.fit(self._score([X[i] for i in idx]), ya[idx])
            self._boot.append(iso)
        if X_val and y_val:
            ps = [self._iso.predict([self._score([x])[0]])[0] for x in X_val]
            self._report = _validation_report([float(p) for p in ps], y_val)
        return self.report()

    def predict_ci(self, x: List[float], txn_id: str | None = None) -> CalibratedP:
        s = float(self._score([x])[0])
        p = float(np.clip(self._iso.predict([s])[0], 0.0, 1.0))
        if not self._boot:
            return CalibratedP(p=p, lo=p, hi=p)
        preds = np.sort(np.asarray(
            [b.predict([s])[0] for b in self._boot], dtype=float))
        lo = float(np.clip(preds[int(0.05 * len(preds))], 0.0, 1.0))
        hi = float(np.clip(preds[int(0.95 * len(preds)) - 1], 0.0, 1.0))
        return CalibratedP(p=p, lo=min(lo, p), hi=max(hi, p))

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "lr_coef": self._lr.coef_.tolist(),
            "lr_intercept": self._lr.intercept_.tolist(),
            "lr_classes": self._lr.classes_.tolist(),
            "X_thresholds": self._iso.X_thresholds_.tolist(),
            "y_thresholds": self._iso.y_thresholds_.tolist(),
            "bootstrap": [
                {"X": b.X_thresholds_.tolist(), "y": b.y_thresholds_.tolist()}
                for b in self._boot
            ],
            "report": self._report,
            "feature_names": FEATURE_NAMES,
        }
        path.write_text(json.dumps(state))

    @classmethod
    def load(cls, path: Path) -> "IsotonicCalibrator":
        state = json.loads(Path(path).read_text())
        cal = cls()
        lr = LogisticRegression(max_iter=1000)
        import numpy as _np
        lr.coef_ = _np.asarray(state["lr_coef"], dtype=float)
        lr.intercept_ = _np.asarray(state["lr_intercept"], dtype=float)
        lr.classes_ = _np.asarray(state["lr_classes"], dtype=int)
        cal._lr = lr
        cal._iso = _new_iso()
        cal._iso.fit(_np.asarray(state["X_thresholds"]),
                     _np.asarray(state["y_thresholds"]))
        cal._boot = []
        for bs in state.get("bootstrap", []):
            iso = _new_iso()
            iso.fit(_np.asarray(bs["X"]), _np.asarray(bs["y"]))
            cal._boot.append(iso)
        cal._report = state.get("report", {})
        return cal

    def report(self) -> Dict[str, Any]:
        return dict(self._report)


def _validation_report(ps: List[float], ys: List[int]) -> Dict[str, Any]:
    """ECE (10 equal-width bins) + Brier + reliability bins for the blog."""
    ps_a = np.asarray(ps, dtype=float)
    ys_a = np.asarray(ys, dtype=float)
    ece = 0.0
    bins: List[Dict[str, Any]] = []
    for i in range(10):
        lo, hi = i / 10.0, (i + 1) / 10.0
        m = (ps_a >= lo) & (ps_a < hi) if i < 9 else (ps_a >= lo) & (ps_a <= hi)
        if m.sum() == 0:
            continue
        conf = float(ps_a[m].mean())
        acc = float(ys_a[m].mean())
        ece += m.sum() / len(ps_a) * abs(conf - acc)
        bins.append({"bin": i, "n": int(m.sum()), "mean_p": round(conf, 4),
                     "fraud_rate": round(acc, 4)})
    brier = float(((ps_a - ys_a) ** 2).mean())
    from sklearn.metrics import roc_auc_score

    try:
        auc = float(roc_auc_score(ys_a, ps_a))
    except ValueError:
        auc = 0.5
    return {"ece": round(ece, 4), "brier": round(brier, 4), "auc": round(auc, 4),
            "n_val": len(ys), "reliability_bins": bins}
