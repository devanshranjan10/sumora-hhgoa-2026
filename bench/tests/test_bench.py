"""Tests 2 and 3: stub runner on all 20 cases (validated), and determinism."""

import json
from pathlib import Path

import pytest

from bench.run import run_benchmark, load_cases, DETERMINISTIC_SEED
from bench.agent import StubAgentRunner
from bench.dataset import DATA_DIR, load_dataset
from bench.validate import validate_answer
from actions_api.personas import ScriptedPersonaResponder

CASES = DATA_DIR / "case_pack.csv"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "personas.json"


@pytest.fixture(scope="module")
def idx():
    return load_dataset(DATA_DIR)


@pytest.fixture(scope="module")
def runner():
    return StubAgentRunner(
        persona=ScriptedPersonaResponder(str(FIXTURES), seed=DETERMINISTIC_SEED),
        seed=DETERMINISTIC_SEED,
    )


def _run(tmp_path: Path, name: str, deterministic: bool, idx, runner) -> Path:
    out = tmp_path / name
    rc = run_benchmark(CASES, out, deterministic=deterministic, runner=runner, idx=idx)
    assert rc == 0, "benchmark run reported validation failures"
    return out


class TestStubRunnerAll20:
    @pytest.fixture(scope="class")
    def out(self, tmp_path_factory, idx, runner):
        return _run(tmp_path_factory.mktemp("bench"), "answers", True, idx, runner)

    def test_twenty_files_written(self, out):
        files = sorted(out.glob("HHG-*.json"))
        assert len(files) == 20
        expected = {c.case_id for c in load_cases(CASES)}
        assert {f.stem for f in files} == expected

    def test_every_file_validates(self, out, idx):
        for f in sorted(out.glob("HHG-*.json")):
            answer = json.loads(f.read_text())
            errors = validate_answer(answer, idx=idx)  # schema+semantics+ids+rule-exists
            assert errors == [], f"{f.name}: {errors}"

    def test_instrumentation_present(self, out):
        for f in sorted(out.glob("HHG-*.json")):
            a = json.loads(f.read_text())
            assert a["tool_calls"] >= 3
            assert a["tokens"] > 0
            assert a["latency_s"] > 0

    def test_customer_report_cases_have_evidence_of_dispute(self, out):
        for f in sorted(out.glob("HHG-*.json")):
            a = json.loads(f.read_text())
            case_row = next(c for c in load_cases(CASES) if c.case_id == a["case_id"])
            if case_row.trigger_type == "customer_report":
                assert a["case"]["verdict"] == "fraud", a["case_id"]
                actions = {x["action"] for x in a["next_best_actions"]["final"]}
                assert {"BLOCK_CARD", "CREATE_CASE"} <= actions  # R2


class TestDeterminism:
    def test_two_deterministic_runs_byte_identical(self, tmp_path, idx):
        out1 = _run(tmp_path, "run1", True, idx, StubAgentRunner(
            persona=ScriptedPersonaResponder(str(FIXTURES), seed=DETERMINISTIC_SEED),
            seed=DETERMINISTIC_SEED))
        out2 = _run(tmp_path, "run2", True, idx, StubAgentRunner(
            persona=ScriptedPersonaResponder(str(FIXTURES), seed=DETERMINISTIC_SEED),
            seed=DETERMINISTIC_SEED))
        files1 = sorted(out1.glob("*.json"))
        files2 = sorted(out2.glob("*.json"))
        assert [f.name for f in files1] == [f.name for f in files2] and len(files1) == 20
        for f1, f2 in zip(files1, files2):
            assert f1.read_bytes() == f2.read_bytes(), f"{f1.name} differs between runs"


class TestIDValidator:
    def test_made_up_txn_id_rejected(self, tmp_path, idx, runner):
        out = _run(tmp_path, "answers", True, idx, runner)
        answer = json.loads((out / "HHG-001.json").read_text())
        answer["case"]["affected_txn_ids"].append("T9999999")
        errors = validate_answer(answer, idx=idx)
        assert any("not in dataset" in e for e in errors)

    def test_made_up_closed_case_rejected(self, idx):
        answer = json.loads(
            (Path(__file__).parents[1] / "fixtures" / "readme_example_answer.json").read_text()
        )
        answer["case"]["similar_prior_cases"] = ["CC-9999"]
        errors = validate_answer(answer, idx=idx)
        assert any("CC-9999" in e for e in errors)

    def test_readme_example_structure_valid(self, idx):
        """The README example is illustrative (its IDs are disguised/fictional),
        so schema + semantic structure must pass; ID existence is expected to
        flag the fictional IDs (covered by the made-up-ID tests)."""
        answer = json.loads(
            (Path(__file__).parents[1] / "fixtures" / "readme_example_answer.json").read_text()
        )
        from bench.validate import validate_schema, validate_semantics

        assert validate_schema(answer) == []
        assert validate_semantics(answer) == []


class TestRuleFiredValidator:
    def test_unknown_rule_rejected(self, tmp_path, idx, runner):
        out = _run(tmp_path, "answers", True, idx, runner)
        answer = json.loads((out / "HHG-001.json").read_text())
        answer["next_best_actions"]["final"][0]["reason"] = "R42: made up rule"
        errors = validate_answer(answer, idx=idx)
        assert any("R42" in e for e in errors)

    def test_unfired_rule_rejected(self, tmp_path, idx, runner):
        out = _run(tmp_path, "answers", True, idx, runner)
        # find a case that produced a legitimate verdict (R3 cited); citing R2
        # there must fail the fired check
        for f in sorted(out.glob("HHG-*.json")):
            answer = json.loads(f.read_text())
            if answer["case"]["verdict"] == "legitimate":
                answer["next_best_actions"]["final"][0]["reason"] = "R2: customer denied"
                state = {"customer_confirmed": True, "verdict": "legitimate"}
                errors = validate_answer(answer, idx=idx, case_state=state)
                assert any("R2" in e and "did not fire" in e for e in errors)
                return
        pytest.skip("no legitimate-verdict case in this run")
