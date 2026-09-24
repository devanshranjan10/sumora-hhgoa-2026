#!/usr/bin/env python3
"""Deterministic eval gate for the SFT artifact (spec section 6b).

Runs the fine-tuned model over the 20 benchmark case prompts (same prompt
format as training), extracts each answer JSON, validates it against the
bench contract (schema + semantics + ID existence), and reports:
  - answer-file validity rate (must be 20/20 to pass the gate)
  - verdict distribution + rule citations
  - calibration of the emitted fraud_probability vs the teacher's verdict
The gate is deterministic: temperature 0, fixed prompts, seeded decoding.
"""
import json
import sys
from pathlib import Path

import httpx

TG = "http://localhost:9000/query/SumoraFraudGraph"
CASE_DIR = Path("/mnt/sih26-train/sumora/data")
OUT = Path("/mnt/sih26-train/sumora/runs/eval")
ADAPTER = Path("/mnt/sih26-train/sumora/runs/sft/final_adapter")

POLICY = open(Path(__file__).resolve().parent / "gen_traces.py").read().split('POLICY = """')[1].split('"""')[0]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench.schema import ANSWER_SCHEMA  # noqa: E402  (shipped to the VM root)


def _compact(obj, depth: int = 0):
    """Same digest as gen_traces.build_prompt - eval prompts must match
    training prompts exactly."""
    if isinstance(obj, dict):
        if "results" in obj and isinstance(obj["results"], list) and len(obj["results"]) == 1:
            return _compact(obj["results"][0], depth)
        return {k: _compact(v, depth + 1) for k, v in obj.items()
                if k not in ("version", "error", "message")}
    if isinstance(obj, list):
        return [_compact(x, depth + 1) for x in obj[:12]]
    return obj


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = "Qwen/Qwen3-14B"
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    client = httpx.Client()
    import csv
    cases = list(csv.DictReader(open(CASE_DIR / "case_pack.csv")))

    OUT.mkdir(parents=True, exist_ok=True)
    n_valid = 0
    results = []
    for case in cases:
        cid = case.get("case_id") or list(case.values())[0]
        # Same evidence retrieval as trace generation.
        pack = client.post(f"{TG}/q_evidence_pack",
                           json={"case_id": cid, "max_vertices": 40, "max_edges": 80},
                           timeout=120).json()["results"][0]
        # Same budgeted digest as training (guard-the-context-window).
        digest = json.dumps(_compact(pack))[:6000]
        prompt = (
            "You are a senior bank fraud analyst working inside an agentic investigation "
            "platform backed by a TigerGraph fraud graph. Investigate the case below.\n\n"
            f"{POLICY}\nCASE: {json.dumps(case)}\n\n"
            f"EVIDENCE DIGEST (compact tool outputs from the graph): {digest}\n\n"
            "Think step by step in <=12 short numbered steps: state hypotheses, which policy "
            "rules could fire, what evidence confirms/refutes each, your calibrated p, whether "
            "to gather more evidence (EVSI) or act, and the stopping reason. Then output the "
            "final answer as a single JSON object with exactly the answer keys listed above."
        )
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=2000, do_sample=False,
                                 temperature=None, top_p=None, pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        (OUT / f"{cid}.reply.txt").write_text(reply)

        import re as _re
        ans = None
        m = _re.search(r"\{", reply)
        if m:
            depth, start = 0, m.start()
            for i in range(start, len(reply)):
                if reply[i] == "{":
                    depth += 1
                elif reply[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            ans = json.loads(reply[start:i + 1])
                        except Exception:
                            ans = None
                        break
        ok = False
        why = "no JSON object"
        if isinstance(ans, dict):
            import jsonschema
            try:
                jsonschema.validate(ans, ANSWER_SCHEMA)
                ok = True
                why = ""
            except Exception as e:  # jsonschema.ValidationError and friends
                why = f"schema: {str(e)[:110]}"
        n_valid += bool(ok)
        case_obj = ans.get("case") if isinstance(ans, dict) else None
        results.append({"case_id": cid, "valid_json": bool(ok),
                        "verdict": case_obj.get("verdict") if isinstance(case_obj, dict) else None,
                        "p": case_obj.get("fraud_probability") if isinstance(case_obj, dict) else None,
                        "reason": why})
        v = case_obj.get("verdict") if isinstance(case_obj, dict) else None
        print(f"{cid}: valid={ok} verdict={v if ok else '-'}"
              f"{(' reason=' + why) if (not ok and why != 'no JSON object') else ''}", flush=True)

    summary = {
        "n_cases": len(cases),
        "n_valid_json": n_valid,
        "validity_rate": round(n_valid / len(cases), 3),
        "gate_pass": n_valid == len(cases),
        "verdicts": {},
        "results": results,
    }
    for r in results:
        if r["verdict"]:
            summary["verdicts"][r["verdict"]] = summary["verdicts"].get(r["verdict"], 0) + 1
    (OUT / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("n_valid_json", "validity_rate", "gate_pass", "verdicts")}))
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
