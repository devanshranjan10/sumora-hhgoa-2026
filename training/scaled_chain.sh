#!/usr/bin/env bash
# ============================================================================
# Sumora scaled chain — runs 24/7 unattended on the A100:
#   wait for scaled trace-gen -> normalize (ground-truth gate) -> QLoRA SFT
#   -> deterministic eval gate -> package GGUF.
#
# Launch:  nohup bash scaled_chain.sh > runs/scaled_chain.log 2>&1 &
# State:   runs/SCALED_CHAIN_STATE = WAIT_GEN | FATAL_* | CHAIN_DONE
# ============================================================================
set -u
S=/mnt/sih26-train/sumora
PY=$S/venv/bin/python
GEN_LOG=$S/traces/scaled_gen.log
STATE=$S/runs/SCALED_CHAIN_STATE

echo WAIT_GEN > "$STATE"
# --- 1. wait for the teacher trace generation to finish (max 5h) ---
for i in $(seq 1 300); do
    if grep -q SCALED_GEN_DONE "$GEN_LOG" 2>/dev/null; then
        break
    fi
    sleep 60
done
if ! grep -q SCALED_GEN_DONE "$GEN_LOG" 2>/dev/null; then
    echo FATAL_GEN_TIMEOUT > "$STATE"; exit 1
fi
RAW=$(wc -l < "$S/traces/traces_scaled_raw.jsonl")
echo "gen done: $RAW raw traces" >&2
if [ "$RAW" -lt 800 ]; then
    echo "FATAL_GEN_TOO_FEW($RAW)" > "$STATE"; exit 1
fi

# --- 2. normalize + ground-truth gate + p anchoring ---
echo NORMALIZING > "$STATE"
$PY $S/normalize_scaled.py > $S/runs/normalize_scaled.log 2>&1
if ! grep -q NORMALIZE_SCALED_DONE $S/runs/normalize_scaled.log; then
    echo FATAL_NORMALIZE > "$STATE"; exit 1
fi
KEPT=$(grep -oE 'kept=[0-9]+' $S/runs/normalize_scaled.log | head -1 | cut -d= -f2)
if [ "${KEPT:-0}" -lt 800 ]; then
    echo "FATAL_NORMALIZE_TOO_FEW($KEPT)" > "$STATE"; exit 1
fi

# --- 3. QLoRA SFT (Qwen3-14B, r=64 a=128) ---
echo TRAINING > "$STATE"
$PY $S/train_sft.py > $S/runs/sft_scaled.log 2>&1
if ! grep -q SFT_DONE $S/runs/sft_scaled.log; then
    echo FATAL_TRAIN > "$STATE"; exit 1
fi
# Real-training gate: first logged train loss must be > 0.05 (0.0 = broken mask).
FIRST_LOSS=$(grep -oE "'loss': [0-9.]+" $S/runs/sft_scaled.log | head -1 | cut -d' ' -f2)
if ! awk -v l="$FIRST_LOSS" 'BEGIN{exit !(l > 0.05)}'; then
    echo "FATAL_LOSS($FIRST_LOSS)" > "$STATE"; exit 1
fi

# --- 4. deterministic eval gate (20 benchmark cases) ---
echo EVALUATING > "$STATE"
$PY $S/eval_sft.py > $S/runs/eval_scaled.log 2>&1
RC=$?
echo "EVAL_RC=$RC" >> $S/runs/eval_scaled.log
# gate_pass recorded either way; packaging proceeds (the artifact is useful
# with its measured limits, per the honest-positioning rule in spec 6b).

# --- 5. package (merge LoRA + q4_k_m GGUF) ---
echo PACKAGING > "$STATE"
$PY $S/package_sft.py > $S/runs/package_scaled.log 2>&1
if grep -qE "PACKAGE_DONE|GGUF written" $S/runs/package_scaled.log; then
    echo CHAIN_DONE > "$STATE"
else
    echo FATAL_PACKAGE > "$STATE"
    exit 1
fi
echo "SCALED CHAIN DONE — see runs/eval_scaled.log + artifacts/" >&2