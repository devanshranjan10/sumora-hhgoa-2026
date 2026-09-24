"""Benchmark runner CLI (spec section 9).

Usage:
    python -m bench.run --cases data/HHGOA_IEEE/case_pack.csv --out answers/ [--deterministic]

For each case: run the agent loop (AgentRunner protocol; StubAgentRunner by
default), instrument tool_calls/tokens/latency, validate the answer against
the section 6a schema + ID-existence + rule-fired checks, and write
answers/<case_id>.json.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List, Optional

from bench.agent import AgentRunner, CaseContext, StubAgentRunner
from bench.dataset import DATA_DIR, DatasetIndex, load_dataset
from bench.instrument import CaseMetrics
from bench.writer import write_answer

DEFAULT_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "personas.json"
DETERMINISTIC_SEED = 42


def load_cases(cases_path: Path) -> List[CaseContext]:
    contexts: List[CaseContext] = []
    with cases_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            contexts.append(
                CaseContext(
                    case_id=row["case_id"],
                    opened_at=row["opened_at"],
                    trigger_type=row["trigger_type"],
                    trigger_text=row["trigger_text"],
                    flagged_txn_id=row["flagged_txn_id"],
                    card_id=row["card_id"],
                    customer_id=row["customer_id"],
                    risk_score=float(row["risk_score"]) if row["risk_score"].strip() else None,
                )
            )
    return contexts


def run_benchmark(
    cases_path: Path,
    out_dir: Path,
    deterministic: bool = False,
    runner: Optional[AgentRunner] = None,
    idx: Optional[DatasetIndex] = None,
    data_dir: Optional[Path] = None,
    graph_agent: bool = False,
    live: bool = False,
) -> int:
    if live and not graph_agent:
        raise ValueError("live=True requires graph_agent=True")
    if deterministic:
        random.seed(DETERMINISTIC_SEED)

    idx = idx or load_dataset(data_dir or DATA_DIR)
    cases = load_cases(cases_path)
    if runner is None and graph_agent:
        from actions_api.personas import ScriptedPersonaResponder

        fixtures = str(DEFAULT_FIXTURES) if DEFAULT_FIXTURES.exists() else None
        from agent.runner import GraphAgentRunner

        runner = GraphAgentRunner(
            persona=ScriptedPersonaResponder(fixtures, seed=DETERMINISTIC_SEED),
            seed=DETERMINISTIC_SEED,
            mode="live" if live else "stub",
        )
    if runner is None:
        from actions_api.personas import ScriptedPersonaResponder

        fixtures = str(DEFAULT_FIXTURES) if DEFAULT_FIXTURES.exists() else None
        runner = StubAgentRunner(
            persona=ScriptedPersonaResponder(fixtures, seed=DETERMINISTIC_SEED),
            seed=DETERMINISTIC_SEED,
        )

    print(f"runner={type(runner).__name__} graph_mode={getattr(runner, 'mode', 'offline')}")
    failures = 0
    for ctx in cases:
        metrics = CaseMetrics()
        metrics.start()
        result = runner.investigate(ctx, idx, metrics)
        answer = result.answer
        # Deterministic mode reports the simulated investigation cost so
        # re-runs are byte-identical; live mode reports wall-clock.
        if deterministic:
            answer["latency_s"] = round(2.5 + 0.35 * metrics.tool_calls, 3)
        else:
            answer["latency_s"] = metrics.latency_s
        answer["tool_calls"] = metrics.tool_calls
        answer["tokens"] = metrics.tokens

        errors = write_answer(
            answer, out_dir, idx=idx, case_state=result.case_state, strict=False
        )
        if errors:
            failures += 1
            print(f"FAIL {ctx.case_id}:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
        else:
            print(
                f"ok   {ctx.case_id}  verdict={answer['case']['verdict']:<10} "
                f"p={answer['case']['fraud_probability']:.2f} "
                f"sar={answer['sar']['file']} "
                f"tools={answer['tool_calls']} tokens={answer['tokens']}"
            )

    print(
        f"\n{len(cases) - failures}/{len(cases)} answer files validated and written to {out_dir}/"
    )
    return 1 if failures else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.run", description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="path to case_pack.csv")
    parser.add_argument("--out", required=True, type=Path, help="answer output directory")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="fixed seed + scripted persona responses; re-runs are byte-identical",
    )
    parser.add_argument(
        "--graph-agent",
        action="store_true",
        help="run the real LangGraph loop (agent/) instead of the stub",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="with --graph-agent: write resolved cases back to the live "
             "TigerGraph (TG_HOST/TG_PORT env; written_to_graph=true)",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    return run_benchmark(args.cases, args.out, args.deterministic,
                         data_dir=args.data_dir, graph_agent=args.graph_agent,
                         live=args.live)


if __name__ == "__main__":
    raise SystemExit(main())
