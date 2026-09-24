#!/usr/bin/env bash
# Morning retrain chain: SFT (normalized Pro traces) -> eval gate -> package.
# State marker in runs/CHAIN_STATE: CHAIN_DONE | FATAL_TRAIN | FATAL_LOSS | FATAL_EVAL
set -u
S=/mnt/sih26-train/sumora
PY=$S/venv/bin/python

$PY $S/train_sft.py > $S/runs/sft.log 2>&1
if ! grep -q SFT_DONE $S/runs/sft.log; then
  echo FATAL_TRAIN > $S/runs/CHAIN_STATE; exit 1
fi
# Real-training gate: first logged train loss must be > 0.05 (0.0 = broken mask)
FIRST_LOSS=$(grep -oE "'loss': [0-9.]+" $S/runs/sft.log | head -1 | cut -d' ' -f2)
if ! awk -v l="$FIRST_LOSS" 'BEGIN{exit !(l > 0.05)}'; then
  echo "FATAL_LOSS($FIRST_LOSS)" > $S/runs/CHAIN_STATE; exit 1
fi

$PY $S/eval_sft.py > $S/runs/eval.log 2>&1
RC=$?
echo "EVAL_RC=$RC" >> $S/runs/eval.log
if [ $RC -ne 0 ]; then
  echo FATAL_EVAL > $S/runs/CHAIN_STATE
  exit 1
fi

$PY $S/package_sft.py > $S/runs/package.log 2>&1
if grep -qE "PACKAGE_DONE|GGUF written" $S/runs/package.log; then
  echo CHAIN_DONE > $S/runs/CHAIN_STATE
else
  echo "FATAL_PACKAGE(no marker)" > $S/runs/CHAIN_STATE
fi
