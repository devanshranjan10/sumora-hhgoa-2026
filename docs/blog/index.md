---
layout: default
title: Sumora technical blog
---

# An agent that can show its fraud evidence

*Sumora's submission for the TigerGraph Agentic Fraud Investigation task at Hacker House Goa.*

[Public code and 20 answer files](https://github.com/devanshranjan10/sumora-hhgoa-2026) ·
[Live investigator](http://34.66.141.236/)

Fraud alerts need evidence before a card is blocked. Sumora joins payment history,
device profiles, closed investigations, and bank policy in TigerGraph Community
Edition. Its LangGraph agent records each tool call, estimates a fraud probability,
and recommends an action with the required approval route.

## One number, owned by the decision model

The deployed agent carries one probability, `p`, from gradient boosting over
as-of account history and the supplied transaction and identity fields. A
six-feature model handles newly uploaded rows that lack those source fields.
The September selection split chooses sigmoid calibration for the rich model
and isotonic calibration for the fallback. Both exclude the bank risk score.

The deterministic explanation layer describes `p`; it never generates it. The
closed-case file has 4,665 confirmed fraud and 900 cleared investigations. All
3,987 cases below bank risk 0.80 are labeled fraud, so the full-cohort model's
apparently strong October result did not transfer to the challenge pack. We
trained a replacement only on the region with both outcomes. Its October
holdout has 295 cases and reached **AUC 0.927, class-balanced Brier 0.109**.
The six-feature fallback reached AUC 0.808 and balanced Brier 0.163 on that
same split. We cap probabilities at 0.05 and 0.95 because the calibration set
has only 125 cases. The hidden challenge labels are unavailable, and these
historical scores do not establish challenge accuracy.

## The gate: gather or act?

Acting wrong costs money; gathering costs delay. The agent computes both. Expected
cost of each action under the bank's cost model, then expected value of sample
information (EVSI): if step-up authentication is likely to change the best
action enough to justify its cost, the agent requests it; otherwise it acts.
HHG-011 and HHG-014 are deliberately left uncertain while that response is
pending. The agent never inserts a fictional response into the evidence record.

## The graph is the memory

Cases, evidence, verdicts and similarity features are written back to TigerGraph.
The next investigation starts from what the bank already confirmed: our signed
memory signal `(n_fraud − n_cleared)/√(n_total + 1)` raised or lowered `p` before
the explanation was written. We report the counterfactual honestly: memory moved
probabilities (mean |Δp| = 0.058, concentrated in the 7/20 cases with prior
history) and flipped one verdict, but a volume-preserving permutation null flips
0–1 verdicts too (p = 0.75). Device memory is real but decision-weak at this
sample size. We said that in the README, because a fraud tool you can't trust to
admit weakness is a fraud tool.

## What TigerGraph actually does

The Community Edition graph holds 590,742 base transactions, 5,565 closed cases,
and the 20 challenge cases. Twenty-six GSQL queries and three graph algorithms
are installed. The live agent queries a bounded evidence pack, prior-case
structural similarity, and pattern scorers, then writes its case, evidence,
decision, and step trace back to TigerGraph. The dashboard renders an interactive
bounded graph neighborhood, node attributes, actual agent-step progress, evidence
citations, approval routes, and recorded/live result comparison.

The pattern scorers and prior-case similarity query run through TigerGraph's
official MCP server, restricted to the installed-query tool on a loopback
listener. Each live trace records the MCP query name, parameters, and a bounded
preview of the returned rows; the UI lets an analyst inspect these receipts.
The bounded evidence pack and case write-back use RESTPP. This keeps MCP in the
live LangGraph investigation without passing a large raw tool catalog to the model.

Analysts can upload 1–5 CSV or JSON rows to create cases for existing graph
transactions or add new transactions with their customer, card, and device edges.
The graph and in-memory index are updated before investigation. New-row estimates
are labeled provisional: the model was trained on historical cases and sparse
customer history is a real limitation.

The prototype runs on one 2-vCPU, 16 GiB VM with TigerGraph, a
Python API, and Nginx. The decision model runs on CPU beside the graph; no GPU
is needed for inference. We tested Spot capacity to target roughly $47.25 per
month including compute, a 100 GB standard disk, and IPv4 before egress and
tax. Repeated preemptions made that unsuitable for the submission window, so
we switched this VM to on-demand capacity; its ongoing monthly cost is higher.

## The sixth pattern (innovation)

Five typologies are documented. We found a sixth with a negative control, not a
vibes label. Statistic: count of Build/-qualified device profiles that link ≥3
distinct cardholders on bank-confirmed fraud cases. Observed 149 rings vs null
mean 126.3 under 2000 label shuffles: **permutation p = 0.0005**. Demo exemplar:
Samsung SM-G935F Chrome/Android (CC-2649 / 2971 / 2985 / 3035), already flagged
by bank analysts as unmatched to the five documented patterns. Name we ship:
**cross-account shared-device ring**. The agent describes it only after the discovery gate.

We also run a time-aware watchlist over transactions outside the 20 scored cases.
It identified **187** November–December candidates with bank risk below 0.50,
a Build-qualified device profile linked to at least three other customers' confirmed
fraud cases that closed before the candidate transaction, and a profile shared by
at most 25 customers. An analyst can open any candidate into TigerGraph and run
the same agent. This is a lead-generation rule, not a claim that those 187 are
fraudulent: device profiles are coarse and can collide across people. Extra
investigations are stored apart from `cases/`.

## The model artifact

Qwen3-14B, QLoRA-fine-tuned on teacher traces, is documented as an experiment. A
strict replay of its saved raw replies fails the full answer contract because the
corpus leaked outcome metadata, dropped action lists, and used a different chat
serialization at training time. The calibrated graph loop remains the decision
maker. We keep the audit report and sample replies; the large weights are not
in the repository.

## Rules you can execute

"Next-best action under policy" usually means a prompt. Ours is code: R1–R10 are
executable guards in the action API, and an answer file's rule citations are
checked against what actually fired. `fired_rules()` is a function, not a claim.
All 20 files in `cases/` validate against schema, semantics, ID-existence, and
rule-fired layers. The committed live run also records `written_to_graph: true`
for all 20 cases.

**Pre-publication audit:** a source-data check found wrong channel claims in
12 earlier scored files and nine incorrect scored card links. We rebuilt the
graph and regenerated all 20 answers against the live graph with the calibrated
gradient-boosting model and raw source fields. The source-fact check reports
zero errors. The new set contains 10 fraud, six legitimate, and four uncertain
verdicts; its hidden-label accuracy remains unverified.

## Limits we can measure

- Our permutation-tested pattern enrichment (2,000 draws) found no statistic
  surviving at α=0.05 with n=20 investigated cases. Directions point the right
  way; the test decides. We ship the p-values.
- Memory's decision impact is within the permutation null. See above.
- The saved fine-tune's reported loss and schema-only decode are diagnostics. The
  full raw-reply replay is the gate, and it reports 0/20 valid answers. We do not
  use that model to choose an action.

The prototype records approval routes and simulates action execution; it does
not connect to a bank's card controls or send regulatory reports. The report
field is a recommendation and draft. The closed-case validation results cannot
resolve the challenge's deliberately different class balance. Our live demo
shows what the agent did and why, including where it remains uncertain.
