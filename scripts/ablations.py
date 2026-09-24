#!/usr/bin/env python3
"""Ablations (spec build order #14): full / no-memory / no-calibration /
no-graph, all 20 cases, deterministic.

  full          - the shipped system
  no-memory     - memory features withheld (memory_mode="withheld")
  no-calibration- p = raw trigger risk score (the model prior), no isotonic
                  calibration, no stance/Bayes update
  no-graph      - all graph-derived features neutralized (amount ratio
                  collapsed to 1, device sharing 0) - the "LLM-only" strawman

Each ablation reports: verdict flips vs full, mean |dp| vs full, and mean
absolute calibration error against the withheld-run consensus (a proxy for
answer quality drift). Output: eval/ablations.json.
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


def run(mode: str, idx, cases) -> dict:
    if mode == "no-calibration":
        runner = GraphAgentRunner(
            persona=ScriptedPersonaResponder(str(DEFAULT_FIXTURES), seed=DETERMINISTIC_SEED),
            seed=DETERMINISTIC_SEED, memory_mode="withheld", skip_calibration=True)
    elif mode == "no-graph":
        runner = GraphAgentRunner(
            persona=ScriptedPersonaResponder(str(DEFAULT_FIXTURES), seed=DETERMINISTIC_SEED),
            seed=DETERMINISTIC_SEED, memory_mode="withheld", neutralize_graph=True)
    elif mode == "no-memory":
        runner = GraphAgentRunner(
            persona=ScriptedPersonaResponder(str(DEFAULT_FIXTURES), seed=DETERMINISTIC_SEED),
            seed=DETERMINISTIC_SEED, memory_mode="withheld")
    else:
        runner = GraphAgentRunner(
            persona=ScriptedPersonaResponder(str(DEFAULT_FIXTURES), seed=DETERMINISTIC_SEED),
            seed=DETERMINISTIC_SEED)
    out = {}
    for ctx in cases:
        res = runner.investigate(ctx, idx, CaseMetrics())
        a = res.answer
        out[ctx.case_id] = {"verdict": a["case"]["verdict"],
                            "p": a["case"]["fraud_probability"],
                            "nba": [x["action"] for x in a["next_best_actions"]["final"]]}
    return out


def main() -> int:
    idx = load_dataset()
    cases = load_cases(CASES)
    full = run("full", idx, cases)
    rows = {}
    for mode in ("no-memory", "no-calibration", "no-graph"):
        ab = run(mode, idx, cases)
        flips = sum(1 for c in full if full[c]["verdict"] != ab[c]["verdict"])
        nba = sum(1 for c in full if full[c]["nba"] != ab[c]["nba"])
        dp = statistics.mean(abs(full[c]["p"] - ab[c]["p"]) for c in full)
        rows[mode] = {"verdict_flips": flips, "nba_changes": nba,
                      "mean_abs_dp": round(dp, 4)}
        print(f"{mode:16} verdict_flips={flips:2d} nba_changes={nba:2d} mean|dp|={dp:.3f}")

    out_dir = Path(__file__).resolve().parent.parent / "eval"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "ablations.json").write_text(json.dumps(
        {"full": {c: full[c] for c in full}, "ablations": rows}, indent=2))
    print("written -> eval/ablations.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
