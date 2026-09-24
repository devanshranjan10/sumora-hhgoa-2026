"""Audit legacy calibrated logistic decisions against the current live graph."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actions_api.personas import ScriptedPersonaResponder
from agent.calibrator import IsotonicCalibrator
from agent.runner import GraphAgentRunner
from bench.dataset import DATA_DIR
from bench.run import DETERMINISTIC_SEED, run_benchmark


def main() -> None:
    idx = joblib.load("/opt/sumora/run/cache/dataset-index.joblib")
    runner = GraphAgentRunner(
        persona=ScriptedPersonaResponder(
            "bench/fixtures/personas.json", seed=DETERMINISTIC_SEED
        ),
        seed=DETERMINISTIC_SEED,
        mode="live",
    )
    runner._cal = IsotonicCalibrator.load(Path("agent/calibrator.json"))
    out = Path("/opt/sumora/run/legacy-answers")
    os.environ["SUMORA_TRACE_DIR"] = "/opt/sumora/run/legacy-traces"
    raise SystemExit(run_benchmark(DATA_DIR / "case_pack.csv", out, runner=runner, idx=idx))


if __name__ == "__main__":
    main()
