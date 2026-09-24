"""The deployed scorer must match the probabilities saved by model selection."""

import json
from pathlib import Path

import pytest

from agent.decision_model import DecisionModel
from agent.features import feature_vector
from bench.dataset import DATA_DIR, load_dataset
from bench.run import load_cases


REPORT = Path(__file__).resolve().parents[2] / "eval/overlap_model/report.json"


@pytest.mark.skipif(not (DATA_DIR / "transactions.csv").exists(),
                    reason="the challenge data is not included in Git")
def test_serving_predictions_match_selected_model_report():
    expected = {
        row["case_id"]: row["p"]
        for row in json.loads(REPORT.read_text())["rich_hgb"]["exam"]
    }
    idx = load_dataset(DATA_DIR)
    model = DecisionModel(data_dir=DATA_DIR)
    for case in load_cases(DATA_DIR / "case_pack.csv"):
        features = feature_vector(idx, case.flagged_txn_id, case.risk_score)["features"]
        actual = model.predict_ci(features, case.flagged_txn_id)
        assert actual.p == pytest.approx(expected[case.case_id], abs=1e-6)
