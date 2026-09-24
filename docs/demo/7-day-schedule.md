# 7-Day Schedule (one week to submission)

Hard rule: core pipeline (steps 1–9) must be green before the fine-tune ships.
The fine-tune is a drop-in via `LLMProvider` — it can never block the submission.

**Day 1 — Graph up.** TigerGraph CE in Docker, schema, loading jobs, 590k txns loaded,
counts verified. Rent A100-80GB spot instance on GCP (keep it stopped until day 4).
*Deliverable: `q_khop_expand` returns a subgraph.*

**Day 2 — Algorithms + MCP.** Louvain/PageRank/cycle-detection installed, pre-computed
to vertex attrs. MCP server connected; agent smoke-calls all 5 queries.
*Deliverable: notebook showing a detected community.*

**Day 3 — Agent loop.** LangGraph state machine, policy engine, approval interrupts,
GraphRAG document path. Teacher LLM drives it.
*Deliverable: one full trace on one case, end to end.*

**Day 4 — Benchmark + trace-gen starts.** Benchmark runner → 20 answer files with
Teacher. Threshold tuning on months 1–3, lock on month 4. START teacher trace-gen on
~1–2k labeled cases (runs overnight). Boot the A100.
*Deliverable: 20 answer files in the exact format + tuning report.*

**Day 5 — Fine-tune.** Filter traces → QLoRA on A100 (2–3 epochs, checkpoints) →
eval gate vs teacher on month 4. Convert winner to GGUF/MLX, verify on this Mac.
*Deliverable: gate report (14B vs teacher) + working local model.*

**Day 6 — Discovery + UI.** Undocumented-pattern discovery. UI polish, demo script
rehearsed, ablations run on the eval VM.
*Deliverable: ablation table + rehearsed demo.*

**Day 7 — Package + ship.** Record demo (local, deterministic mode), write blog post,
final answer-file regeneration with the winning model, social post, submission.
*Deliverable: SUBMITTED.*

## Fallback triggers
- Day 4 EOD: no answer files → descope UI to a CLI demo; answer files are mandatory.
- Day 5 EOD: 14B loses the gate badly → demo on Teacher, fine-tune = "future work"
  blog section. Still a complete submission.
- Day 6 EOD: pattern discovery hasn't found anything → ship it as "agent flags
  anomalous communities for analyst review" (honest framing, still novel).
