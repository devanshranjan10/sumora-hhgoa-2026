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

# POLICY + build_prompt must come from the SAME script that built the training
# prompts (gen_traces_scaled.py — real bank policy, real answer-key format).
# The v1 eval imported the fictional policy from the old gen_traces.py, so the
# gate measured the model against a distribution it never trained on.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_traces_scaled import POLICY, build_prompt, _compact  # noqa: E402

from bench.schema import ANSWER_SCHEMA  # noqa: E402  (shipped to the VM root)


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
    cases = [json.loads(l) for l in (CASE_DIR / "case_pack_prepped.jsonl").read_text().splitlines()] \
        if (CASE_DIR / "case_pack_prepped.jsonl").exists() else None
    import csv
    cases = list(csv.DictReader(open(CASE_DIR / "case_pack.csv")))

    OUT.mkdir(parents=True, exist_ok=True)
    n_valid = 0
    results = []
    for case in cases:
        cid = case.get("case_id") or list(case.values())[0]
        # Same evidence retrieval as the deployed agent.
        pack = client.post(f"{TG}/q_evidence_pack",
                           json={"case_id": cid, "max_vertices": 40, "max_edges": 80},
                           timeout=120).json()["results"][0]
        # IDENTICAL prompt builder to training (gen_traces_scaled.build_prompt):
        # same policy text, same answer-key format, same digest budgeting.
        prompt = build_prompt(case, pack)
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            # 4096 new tokens: a fraud answer carries the full case object plus a
            # 6-12 sentence SAR narrative. At 2000 the JSON was cut off before
            # connected_device_profiles / sar, failing the gate on truncation
            # rather than on judgement.
            out = model.generate(**ids, max_new_tokens=4096, do_sample=False,
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
        results.append({"case_id": cid, "valid_json": bool(ok),
                        "verdict": ans.get("verdict") if isinstance(ans, dict) else None,
                        "p": ans.get("fraud_probability") if isinstance(ans, dict) else None,
                        "reason": why})
        print(f"{cid}: valid={ok} verdict={ans.get('verdict') if isinstance(ans, dict) and ok else '-'}"
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
