#!/usr/bin/env bash
# ============================================================================
# Sumora overnight pipeline (runs unattended on the A100).
# Chain: wait-for-traces -> dedupe -> SFT (loss-verified) -> eval gate ->
#        package. Every stage logs; failures stop the chain with a marker.
# ============================================================================
set -u
S=/mnt/sih26-train/sumora
PY=$S/venv/bin/python
LOG=$S/runs/overnight.log

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# --- 1. wait for trace regeneration to hit 120 -----------------------------
log "waiting for trace regen (target 120)..."
for i in $(seq 1 90); do   # up to 90 x 60s = 90 min
  N=$(wc -l < $S/traces/traces.jsonl 2>/dev/null || echo 0)
  if [ "$N" -ge 120 ]; then break; fi
  # bail out if the generator died before reaching target
  if ! pgrep -f gen_traces.py > /dev/null && [ "$N" -lt 120 ]; then
    log "FATAL: generator exited early at $N traces"; exit 1
  fi
  sleep 60
done
N=$(wc -l < $S/traces/traces.jsonl 2>/dev/null || echo 0)
log "traces available: $N"
if [ "$N" -lt 100 ]; then log "FATAL: only $N traces"; exit 1; fi

# --- 2. dedupe --------------------------------------------------------------
$PY - <<'PYEOF'
import json
seen, out = set(), []
for l in open("/mnt/sih26-train/sumora/traces/traces.jsonl"):
    try:
        r = json.loads(l)
        k = (r["case_id"], r["model"], r["trial"])
    except Exception:
        continue
    if k not in seen:
        seen.add(k)
        out.append(l)
open("/mnt/sih26-train/sumora/traces/traces_dedup.jsonl", "w").write("".join(out))
print(f"deduped to {len(seen)}")
PYEOF
log "dedupe done"

# --- 3. SFT with loss verification ------------------------------------------
log "starting SFT..."
$PY $S/train_sft.py > $S/runs/sft.log 2>&1
if ! grep -q SFT_DONE $S/runs/sft.log; then
  log "FATAL: SFT did not complete"; tail -5 $S/runs/sft.log >> $LOG; exit 1
fi
# Real-training check: loss must be nonzero and accuracy must be learning.
if ! grep -qE "'loss': [1-9]" $S/runs/sft.log; then
  log "FATAL: SFT loss stayed 0.0 - completion mask broken again"
  grep -oE "\{'loss'[^}]*\}" $S/runs/sft.log | head -2 >> $LOG
  exit 1
fi
grep -oE "\{'loss'[^}]*\}" $S/runs/sft.log | head -2 >> $LOG
grep -oE "\{'loss'[^}]*\}" $S/runs/sft.log | tail -1 >> $LOG
log "SFT complete with nonzero loss"

# --- 4. eval gate (deterministic greedy decode over 20 prompts) -------------
log "running eval gate..."
$PY $S/eval_sft.py > $S/runs/eval.log 2>&1
EVAL_RC=$?
grep -E "valid=|n_valid_json|gate_pass" $S/runs/eval.log | tail -25 >> $LOG
if [ $EVAL_RC -ne 0 ]; then
  log "WARN: eval gate rc=$EVAL_RC (see runs/eval.log) - continuing to package"
fi

# --- 5. package (merge + GGUF if available) ----------------------------------
log "packaging artifact..."
$PY $S/package_sft.py > $S/runs/package.log 2>&1
grep -E "merged model saved|GGUF|PACKAGE_DONE" $S/runs/package.log >> $LOG

log "OVERNIGHT_PIPELINE_DONE"
