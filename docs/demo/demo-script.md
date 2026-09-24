# Demo recording: one continuous take, about 4 minutes 40 seconds

Record the browser at 1080p with your voice and cursor. Keep the entire recording
unedited and between 3 and 5 minutes. Speak at a normal pace and let both live
investigations finish on screen. The times below are guides; spend the spare
seconds on the actual tool results, not a title screen.

Before recording, open these tabs in this order so switching tabs does not
interrupt the take:

1. [Live investigator](http://34.66.141.236/)
2. [Runtime architecture](https://github.com/devanshranjan10/sumora-hhgoa-2026/blob/main/docs/architecture.md)
3. [LangGraph flow in `agent/runner.py`](https://github.com/devanshranjan10/sumora-hhgoa-2026/blob/main/agent/runner.py#L331-L364)
4. [Official MCP call in `agent/mcp_graph.py`](https://github.com/devanshranjan10/sumora-hhgoa-2026/blob/main/agent/mcp_graph.py#L20-L53)
5. [Evidence-vs-action gate in `agent/gate.py`](https://github.com/devanshranjan10/sumora-hhgoa-2026/blob/main/agent/gate.py#L145-L208)
6. [The 20 answer files](https://github.com/devanshranjan10/sumora-hhgoa-2026/tree/main/cases)
7. [Model and graph evaluation](https://github.com/devanshranjan10/sumora-hhgoa-2026/blob/main/docs/evaluation.md)

Check that the first tab says **Graph + MCP + model online**. HHG-013 and
HHG-011 should be visible in the left case list. If the VM is warming up, wait
for that status before starting. Close notifications and avoid showing account
details. Do one practice pass without recording so you know where the Agent
activity panel and the open MCP result are on your screen.

| Time | Show and click | Say, in your own voice |
|---|---|---|
| 0:00–0:25 | Start on the live UI. Point to **Submission cases 20**, **27 total**, the service status, and **590,747 transactions · 5,565 closed cases**. | “This is Sumora, our TigerGraph fraud investigator. Twenty cases are scored; seven more are demo imports. The live graph has 590,747 transactions, including five new demo rows, and 5,565 closed investigations. TigerGraph Community Edition, its MCP server, and the decision model run together on a CPU VM.” |
| 0:25–1:00 | Select **HHG-013**, open **Graph**, and click the flagged transaction and a linked card or device. Let the node attributes appear. | “A case starts with a real graph neighborhood: transaction, card, account, device, and earlier cases. These nodes and source attributes come from TigerGraph. A shared link is evidence to investigate, not an automatic fraud label.” |
| 1:00–1:45 | Return to **Overview** and click **Investigate live** for HHG-013. Wait for completion. Open **Agent activity**, scroll to **TigerGraph MCP · captured response**, and point to the open `q_new_device_cnp_score` result. | “This is a fresh run, not playback. The agent reads graph and policy evidence, calls installed GSQL scorers through the official TigerGraph MCP tool, and records the returned rows. Here are the query name, parameters, and actual response. It estimates nine percent fraud probability and recommends monitoring and closing this case as no fraud.” |
| 1:45–2:45 | Select **HHG-011**, click **Investigate live**, then **Agent activity**. Show the `gather` step and another MCP response. Open **Evidence** to show **Response status: No response supplied; request pending**. Return to **Overview** to show **Step Up Auth**, **Uncertain**, and **Open**. | “This second case is ambiguous. Its estimate is 38 percent, and the cost gate finds step-up authentication worth requesting before a fraud decision. The agent makes the request and stops. No customer response was supplied, so it does not invent one or force a verdict. The case stays open with monitoring while the response is pending.” |
| 2:45–3:15 | Show **Watchlist**, **Add data**, and **New typology** briefly. Do not start an import during the recording. | “Outside the scored pack, a time-aware graph scan has 186 unimported leads. Analysts can submit existing graph transaction IDs or entirely new transaction rows. We also found a cross-account shared-device ring outside the five given patterns; a 2,000-shuffle test gave p = 0.0005. These are leads, not hidden fraud labels.” |
| 3:15–4:10 | Switch to the prepared code tabs. On `docs/architecture.md`, point to the component table. On `agent/runner.py`, point to the `StateGraph` nodes and conditional gate. Briefly show `agent/mcp_graph.py` and `agent/gate.py`. | “The browser talks to the Python API, never directly to the database. LangGraph owns the investigation loop. Its conditional gate either gathers evidence or proposes a policy action. `mcp_graph.py` calls TigerGraph's installed-query MCP tool; the agent records the returned rows. The calibrated gradient-boosting model supplies one probability, while `gate.py` compares expected action cost with the value of more evidence. GraphRAG retrieves citable policy text. Policy guards, approval routes, and case write-back are code, not claims in a prompt.” |
| 4:10–4:45 | Show the `cases/` folder, then `docs/evaluation.md` near **Replacement decision model**. Finish on the public README or live UI. | “All 20 answer files are at the repo root under `cases/`, with validated graph write-back. Our predictions are ten fraud, six legitimate, and four uncertain; those are not known outcomes. On 295 historical October cases from the overlap where both labels exist, the model reached AUC 0.927 and class-balanced Brier 0.109. The hidden challenge labels are unavailable. The README explains how to download the dataset, load TigerGraph, and rerun everything.” |

Keep moving if a run takes longer than the guide: describe the step currently
visible rather than claiming a result before it appears. If the service fails,
restart the take after it recovers. **Replay recorded trace** is useful for
practice but does not replace the required live runs. The agent recommends bank
actions; it does not operate real card controls or file reports. The Qwen3-14B
fine-tune is an audited experiment, not the deployed decision model.

After recording, upload the video with **Anyone with the link** access. Open
the exact URL in a private browser window before putting it in the form. Do
not edit `cases/` after the final form submission.

## Numbers to keep accurate

| Claim | Verified value | What it means |
|---|---:|---|
| Scored answer files | 20 | `cases/HHG-001.json` through `HHG-020.json` |
| Cases visible in the app | 27 | 20 scored plus seven demo/imported cases |
| Predicted verdicts | 10 fraud · 6 legitimate · 4 uncertain | Predictions, not challenge accuracy |
| Pending evidence requests | HHG-011 and HHG-014 | Both request step-up authentication; no response is fabricated |
| Graph transactions | 590,747 | 590,742 source rows plus five demo imports |
| Closed historical cases | 5,565 | Supplied investigations, not scored labels |
| Within-card `NEXT` edges | 575,896 | 575,892 prepared plus four demo edges |
| Unimported watchlist leads | 186 | One additional lead was opened as HHG-902 |
| Graph census | 14/14 passed | Live TigerGraph verification |
| Answer validity | 20/20 passed | Schema, semantics, IDs, rules, source facts, graph write-back, and trace parity |
| Historical October overlap | AUC 0.927; balanced Brier 0.109; n = 295 | Selected closed-case holdout, not hidden challenge performance |
| Discovered ring | permutation p = 0.0005; 2,000 shuffles | Lead-generation pattern, not a verdict |
