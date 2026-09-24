#!/usr/bin/env python3
"""Quarantine truncated/unparseable scaled traces so gen can regenerate them.

Root cause: the first scaled run used maxOutputTokens=4096, which truncated the
JSON for ~65% of fraud cases (long SAR narratives). Those records sit in
traces_scaled_raw.jsonl and would be skipped by the resume logic, so they must
be moved out first; gen_traces_scaled.py then regenerates exactly the missing
(case_id, trial) keys at 8192 tokens.

Run on the A100:  ./venv/bin/python prune_truncated.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/sih26-train/sumora")
from normalize_traces import extract_last_json  # noqa: E402

S = Path("/mnt/sih26-train/sumora/traces")
RAW = S / "traces_scaled_raw.jsonl"
BAD = S / "traces_truncated.jsonl"
BAK = S / "traces_scaled_raw.jsonl.preprune"


def main() -> int:
    recs = [json.loads(l) for l in RAW.read_text().splitlines() if l.strip()]
    good, bad = [], []
    for r in recs:
        try:
            ans = extract_last_json(r["messages"][1]["content"])
        except Exception:
            ans = None
        (good if isinstance(ans, dict) and "case_id" in ans else bad).append(r)

    shutil.copy2(RAW, BAK)
    RAW.write_text("".join(json.dumps(r) + "\n" for r in good))
    BAD.write_text("".join(json.dumps(r) + "\n" for r in bad))

    from collections import Counter
    print(f"total={len(recs)} kept={len(good)} quarantined={len(bad)}")
    print("quarantined by outcome:",
          dict(Counter(r.get("outcome") for r in bad)))
    print("kept by outcome:", dict(Counter(r.get("outcome") for r in good)))
    print(f"backup at {BAK}")
    print("PRUNE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
