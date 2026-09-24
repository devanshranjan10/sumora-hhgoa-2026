#!/usr/bin/env python3
"""Validate SFT eval replies against the full bench contract (local).

Pulls runs/eval/*.reply.txt from the A100 (or reads a local dir), extracts the
answer JSON with the same scan as eval_sft.py, then runs bench.validate_answer
(schema + semantics + ID existence against the real dataset index). Reports the
verdict distribution - the modal-collapse test: BOTH fraud and legitimate must
appear for the gate to mean anything.

Usage:
  python scripts/validate_replies.py            # scp from sih26-a100
  python scripts/validate_replies.py --dir DIR  # local replies dir
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bench.dataset import load_dataset  # noqa: E402
from bench.validate import validate_answer  # noqa: E402

VM = "sih26-a100"
ZONE = "us-central1-a"
PROJ = "contral-6b0bd"
VM_DIR = "/mnt/sih26-train/sumora/runs/eval"


def pull_replies(dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["gcloud", "compute", "scp", "--recurse",
         f"{VM}:{VM_DIR}", str(dest.parent / "eval_pull"),
         "--zone", ZONE, "--project", PROJ],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-500:], file=sys.stderr)
        return 1
    src = dest.parent / "eval_pull" / "eval"
    n = 0
    for f in src.glob("*.reply.txt"):
        (dest / f.name).write_bytes(f.read_bytes())
        n += 1
    return n


def extract_answer(reply: str):
    """First balanced top-level {...} that parses as JSON (same as eval_sft)."""
    m = reply.find("{")
    if m < 0:
        return None
    depth, start = 0, m
    for i in range(start, len(reply)):
        if reply[i] == "{":
            depth += 1
        elif reply[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(reply[start:i + 1])
                except Exception:
                    return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=None)
    args = ap.parse_args()

    if args.dir:
        rdir = args.dir
    else:
        rdir = Path(tempfile.mkdtemp(prefix="sft_replies_"))
        n = pull_replies(rdir)
        if n == 0:
            print("no replies pulled", file=sys.stderr)
            return 1
        print(f"pulled {n} replies -> {rdir}")

    idx = load_dataset()
    files = sorted(rdir.glob("*.reply.txt"))
    verdicts: Counter[str] = Counter()
    ps: list[float] = []
    sars = 0
    n_valid = 0
    report = []
    for f in files:
        cid = f.name.split(".")[0]
        ans = extract_answer(f.read_text())
        row = {"case_id": cid, "valid": False, "verdict": None, "p": None}
        if ans is None:
            row["reason"] = "no parseable JSON"
        else:
            errs = validate_answer(ans, idx=idx)
            case = ans.get("case") or {}
            row["verdict"] = case.get("verdict")
            row["p"] = case.get("fraud_probability")
            row["valid"] = not errs
            row["reason"] = "; ".join(errs[:2]) if errs else ""
            if row["verdict"]:
                verdicts[row["verdict"]] += 1
                ps.append(float(row["p"]))
            if (ans.get("sar") or {}).get("file"):
                sars += 1
        n_valid += bool(row["valid"])
        report.append(row)
        flag = "ok" if row["valid"] else f"FAIL {row['reason'][:90]}"
        print(f"{cid}: {row['verdict']} @ {row['p']}  [{flag}]")

    print("\n=== DISTRIBUTION (collapse test) ===")
    print("valid:", f"{n_valid}/{len(files)}")
    print("verdicts:", dict(verdicts))
    if ps:
        ps.sort()
        print(f"p: min={ps[0]} med={ps[len(ps)//2]} max={ps[-1]}")
    print("sar filed:", sars)
    collapsed = len(verdicts) < 2 and len(files) >= 10
    print("modal collapse:", "YES - distribution has <2 verdicts" if collapsed
          else "NO - multiple verdicts present")

    out = REPO / "eval" / "sft" / "scaled_replies_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_files": len(files), "n_valid": n_valid,
        "verdicts": dict(verdicts), "sar_filed": sars,
        "modal_collapse": collapsed, "results": report}, indent=2))
    print(f"wrote {out}")
    return 0 if not collapsed else 2


if __name__ == "__main__":
    raise SystemExit(main())
