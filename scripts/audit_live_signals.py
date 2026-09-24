"""Print bounded live graph scorer outputs for the 20 submitted cases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.tools import InvestigationTools
from bench.dataset import DATA_DIR
from bench.run import load_cases


def main() -> None:
    idx = joblib.load("/opt/sumora/run/cache/dataset-index.joblib")
    for case in load_cases(DATA_DIR / "case_pack.csv"):
        tools = InvestigationTools(idx, mode="live")
        result = {
            "case_id": case.case_id,
            "trigger": case.trigger_type,
            "risk": idx.txn_risk.get(f"T{case.flagged_txn_id}"),
            "online": idx.txn_online.get(f"T{case.flagged_txn_id}"),
            "card_testing": tools.q_card_testing_score(
                case.card_id, case.flagged_txn_id
            ).data,
            "new_device_cnp": tools.q_new_device_cnp_score(case.flagged_txn_id).data,
            "out_of_region": tools.q_out_of_region_score(case.flagged_txn_id).data,
            "ato": tools.q_ato_score(case.customer_id).data,
            "similar": tools.q_case_structural_similarity(
                idx.txn_device.get(f"T{case.flagged_txn_id}", ""),
                case_id=case.case_id,
            ).data,
        }
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
