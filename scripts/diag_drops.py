#!/usr/bin/env python3
"""Diagnose dropped traces: prose-only vs malformed fenced JSON."""
import json
import re

n_fence = n_nofence = n_parsefail = 0
fail_samples = []
model_split = {"pro_fail": 0, "pro_ok": 0, "flash_fail": 0, "flash_ok": 0}
for line in open("/mnt/sih26-train/sumora/traces/traces_dedup.jsonl"):
    rec = json.loads(line)
    comp = rec["messages"][1]["content"]
    model = rec.get("model") or ""
    key = "pro" if "pro" in model else "flash"
    if "```json" in comp or re.search(r'\{\s*"case_id"', comp):
        n_fence += 1
        found = False
        for i in range(len(comp) - 1, -1, -1):
            if comp[i] != "{":
                continue
            depth = 0
            for j in range(i, len(comp)):
                if comp[j] == "{":
                    depth += 1
                elif comp[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            json.loads(comp[i:j + 1])
                            found = True
                        except Exception:
                            pass
                        break
                if found:
                    break
            if found:
                break
        if found:
            model_split[key + "_ok"] += 1
        else:
            n_parsefail += 1
            model_split[key + "_fail"] += 1
            if len(fail_samples) < 2:
                m = comp.rfind("```json")
                fail_samples.append(comp[m:m + 400] if m >= 0 else comp[-400:])
    else:
        n_nofence += 1
        model_split[key + "_fail"] += 1

print("fenced:", n_fence, "parse-fail:", n_parsefail, "prose-only:", n_nofence)
print("model split:", json.dumps(model_split))
for s in fail_samples:
    print("---")
    print(s)
