#!/usr/bin/env bash
# Morning status: one command for the overnight pipeline verdict.
#   bash scripts/morning_status.sh
set -u
HOST="sih26-a100"
ZONE="us-central1-a"
PROJ="contral-6b0bd"
S=/mnt/sih26-train/sumora

gcloud compute ssh "$HOST" --zone="$ZONE" --project="$PROJ" --command="
echo '=== MARKER'; cat $S/runs/MORNING_READY 2>/dev/null || echo 'pipeline still running'
echo '=== OVERNIGHT LOG'; tail -8 $S/runs/overnight.log
echo '=== SFT (fresh run only)'; grep -a 'traces:' $S/runs/sft.log 2>/dev/null | tail -1; grep -aoE \"\\{'loss'[^}]*\\}\" $S/runs/sft.log 2>/dev/null | head -1; grep -aoE \"\\{'loss'[^}]*\\}\" $S/runs/sft.log 2>/dev/null | tail -1; grep -c SFT_DONE $S/runs/sft.log 2>/dev/null
echo '=== EVAL'; grep -aE 'valid=|GATE|gate' $S/runs/eval.log 2>/dev/null | tail -6
echo '=== PACKAGE'; tail -3 $S/runs/package.log 2>/dev/null
echo '=== ARTIFACTS'; ls -lh $S/artifacts/ 2>/dev/null
echo '=== GPU'; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo '=== TRACES'; wc -l $S/traces/traces.jsonl 2>/dev/null"
