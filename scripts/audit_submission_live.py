"""Re-run all 20 cases through the deployed API without touching cases/."""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BASE = "http://127.0.0.1:8000"
OUT = Path("/opt/sumora/run/audit-submission.json")


def request(path: str, method: str = "GET") -> dict:
    with urlopen(Request(BASE + path, method=method), timeout=30) as response:
        return json.load(response)


def main() -> None:
    results = []
    for number in range(1, 21):
        case_id = f"HHG-{number:03d}"
        try:
            run_id = request(f"/api/cases/{case_id}/runs", "POST")["run_id"]
            deadline = time.monotonic() + 180
            while True:
                run = request(f"/api/runs/{run_id}")
                if run["status"] != "running":
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{case_id} exceeded 180 seconds")
                time.sleep(0.5)
            if run["status"] != "complete":
                raise RuntimeError(run.get("error", "Unknown run failure"))
            answer = run["answer"]
            case = answer["case"]
            row = {
                "case_id": case_id,
                "verdict": case["verdict"],
                "probability": case["fraud_probability"],
                "pattern": case["pattern"],
                "flagged_claim": case["evidence"][0]["claim"],
                "prior_cases": len(case["similar_prior_cases"]),
                "actions": [action["action"] for action in answer["next_best_actions"]["final"]],
                "latency_s": answer["latency_s"],
            }
        except (HTTPError, OSError, ValueError, KeyError, RuntimeError, TimeoutError) as exc:
            row = {"case_id": case_id, "error": str(exc)}
        results.append(row)
        OUT.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
