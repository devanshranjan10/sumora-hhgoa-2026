# Autonomous alert investigation

`watchlist.json` is the deterministic top-20 view of 187 candidate transactions
found outside the scored case pack. The rule requires an exam-period transaction
with bank risk below 0.50, a Build-qualified device profile seen on no more than
25 customers, and earlier confirmed fraud on at least three *other* customers
using that profile. “Earlier” is checked against each candidate's timestamp, so
closed cases from the future cannot supply evidence. The profile is coarse and
may be shared by unrelated people; this is an alert queue, not ground truth.

The first alert, `T3425188`, had a bank risk score of 0.08 and was opened as
`HHG-902` through the live dashboard. Its graph case was written to TigerGraph.
`HHG-902.json` and `HHG-902.trace.json` are the resulting answer and
agent audit trail. The model estimate is **not** an observed fraud label;
the recorded actions are recommendations with human approval routes.

Reproduce the watchlist locally after preparing the challenge data:

```bash
python -c 'from bench.dataset import load_dataset,norm_txn_id,DATA_DIR; from bench.run import load_cases; from service.watchlist import build_watchlist; idx=load_dataset(); print(build_watchlist(idx,{norm_txn_id(c.flagged_txn_id) for c in load_cases(DATA_DIR/"case_pack.csv")})["matching_transactions"])'
```

The source resolver derives card IDs from explicit case links and exact card
fingerprints. The watchlist transaction resolves to `C08962-K2` in the rebuilt
TigerGraph. The 20 files in `cases/` are unchanged by the extra investigation.
