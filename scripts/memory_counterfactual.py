#!/usr/bin/env python3
"""Memory counterfactual experiment (spec §6 headline metric) - permutation form.

Design:
  REAL      - memory features active (mem_signal from bank-confirmed cases)
  WITHHELD  - memory zeroed (the no-memory baseline)
  SHUFFLED  - K permutation draws (device->direction association broken,
              per-device volume preserved); each draw is "memory-shaped
              noise" matched to real memory's magnitude distribution

Decision-level readout (the spec's claim is about decisions):
  changed_real   = #cases whose verdict/NBA differ REAL vs WITHHELD
  changed_null   = distribution of #changes across K shuffled draws
  p_value        = fraction of draws with changes >= changed_real
  dp_null_p95    = 95th pct of mean |dp| across draws vs REAL-memory |dp|

The key property vs the previous single-draw version: per-device volume is
preserved in every draw, so the null is matched in shape, not just scale.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.run import DEFAULT_FIXTURES, DETERMINISTIC_SEED, load_cases  # noqa: E402
from actions_api.personas import ScriptedPersonaResponder  # noqa: E402
from agent.runner import GraphAgentRunner  # noqa: E402
from bench.dataset import load_dataset  # noqa: E402
from bench.instrument import CaseMetrics  # noqa: E402

CASES = Path(__file__).resolve().parent.parent / "data" / "HHGOA_IEEE" / "prepped" / "case_pack.csv"
K_DRAWS = 20


def run_mode(mode: str, idx, cases) -> dict:
    runner = GraphAgentRunner(
        persona=ScriptedPersonaResponder(str(DEFAULT_FIXTURES), seed=DETERMINISTIC_SEED),
        seed=DETERMINISTIC_SEED, memory_mode=mode,
    )
    out = {}
    for ctx in cases:
        res = runner.investigate(ctx, idx, CaseMetrics())
        a = res.answer
        out[ctx.case_id] = {
            "verdict": a["case"]["verdict"],
            "p": a["case"]["fraud_probability"],
            "nba": [x["action"] for x in a["next_best_actions"]["final"]],
            "sar": a["sar"]["file"],
        }
    return out


def diff(a: dict, b: dict) -> dict:
    verdicts = sum(1 for cid in a if a[cid]["verdict"] != b[cid]["verdict"])
    nba = sum(1 for cid in a if a[cid]["nba"] != b[cid]["nba"])
    dp = statistics.mean(abs(a[cid]["p"] - b[cid]["p"]) for cid in a)
    return {"verdict_changes": verdicts, "nba_changes": nba, "mean_abs_dp": round(dp, 4)}


def main() -> int:
    idx = load_dataset()
    cases = load_cases(CASES)

    real = run_mode("full", idx, cases)
    withheld = run_mode("withheld", idx, cases)
    d_real = diff(real, withheld)

    draws = []
    for k in range(K_DRAWS):
        shuf = run_mode(f"shuffled:{42 + k}", idx, cases)
        draws.append(diff(shuf, withheld))
    null_verdict = sorted(d["verdict_changes"] for d in draws)
    null_dp = sorted(d["mean_abs_dp"] for d in draws)
    p_value = sum(1 for v in null_verdict if v >= d_real["verdict_changes"]) / len(draws)
    dp_p95 = null_dp[int(0.95 * len(null_dp)) - 1]

    summary = {
        "n_cases": len(cases),
        "k_permutation_draws": K_DRAWS,
        "real_vs_withheld": d_real,
        "null_distribution_verdict_changes": null_verdict,
        "null_mean_abs_dp_p95": dp_p95,
        "permutation_p_value": p_value,
        "rows_real": {cid: real[cid] for cid in real},
        "headline": (
            f"Memory changed the verdict in {d_real['verdict_changes']}/20 cases "
            f"(NBA in {d_real['nba_changes']}/20), mean |dp| = "
            f"{d_real['mean_abs_dp']:.3f}; shuffled-memory null: "
            f"{null_verdict} changes across {K_DRAWS} draws, p = {p_value:.2f}."
        ),
    }
    out = Path(__file__).resolve().parent.parent / "eval"
    out.mkdir(exist_ok=True)
    (out / "memory_counterfactual.json").write_text(json.dumps(summary, indent=2))
    print(summary["headline"])
    print("null draws (verdict changes):", null_verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
