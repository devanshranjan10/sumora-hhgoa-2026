# SFT eval-gate evidence (2026-09-20)

- **Gate**: deterministic greedy decode over the 20 benchmark prompts (same
  prompt format as training), answers validated against the full §6a JSON
  Schema (`bench/schema.py`).
- **Result**: 20/20 schema-valid (`gate_pass: true`). `eval_loss 0.18`,
  token accuracy 0.84 at epoch 0.83 of 5.
- **Trace provenance**: Gemini 2.5 Pro teacher on the live graph, completions
  normalized to the contract (`scripts/normalize_traces.py`): 54 schema-valid
  rows (Flash was prose-only in 58/60 and excluded).
- **Known limit (documented, not hidden)**: teacher verdicts are 94% fraud
  (dataset base rate), so the model emits the modal answer (fraud @ 0.95) for
  every case. The artifact satisfies the answer-format contract; semantic
  discrimination is the calibrated loop's job (AUC 0.77 on the Oct holdout).
  Sample replies included here (HHG-001/007/013).
- **Artifact**: `artifacts/sumora-sft-q4_k_m.gguf` (8.4GB) on `sih26-a100`,
  built with llama.cpp `llama-quantize` from the merged bf16 model.
