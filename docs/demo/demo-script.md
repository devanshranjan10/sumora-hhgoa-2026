# Four-minute demo recording

Record one continuous 1080p screen capture with your own voice. Do not cut or
speed up the footage. Before starting, confirm the live URL opens in a private
browser window, the status rail says **Graph + MCP + model online**, and the
**Graph** tab returns nodes. Upload the finished video with access set to
**Anyone with the link**. Open its URL in a private window before putting it
in the submission form.

| Time | On screen | Say |
|---|---|---|
| 0:00–0:30 | Live workspace and scored-case list | “These 20 cases are the scored answer files in `cases/`. The model runs on this CPU VM beside TigerGraph Community Edition. The hidden labels are unavailable.” |
| 0:30–1:05 | **Graph** for HHG-013; click a transaction, card, and device | “The graph comes from TigerGraph, not a static diagram. It links the flagged transaction to card, account, device, and prior evidence. The selected node shows its source attributes.” |
| 1:05–1:55 | Select HHG-013 and click **Investigate live**; open **Agent activity** | “The agent gathers a bounded evidence pack, calls installed GSQL scorers through TigerGraph MCP, estimates fraud probability without using the supplied bank risk score, and applies the bank’s action policy. This case closes as legitimate.” Expand an **MCP evidence** receipt to show the tool name, query parameters, and returned row preview. Let the action and approval route appear on screen. |
| 1:55–2:55 | Select HHG-011 and click **Investigate live**; open **Agent activity** and its MCP evidence | “This case remains uncertain. The gather-or-act gate estimates that step-up authentication is worth its short delay. The agent requests it and stops for a real response; it does not invent one or force a fraud verdict.” Point to the pending evidence request, unchanged probability, and recommended step-up action. |
| 2:55–3:35 | **Add data** and **Watchlist** | “We can investigate an existing graph transaction or import a new transaction package. The watchlist proposes leads outside the 20 scored cases using earlier confirmed graph links, then sends each lead through the same investigation.” Show the two import formats and open a watchlist graph. |
| 3:35–4:20 | **New typology**, then public repository `cases/`, README, and architecture diagram | “The sixth typology is a cross-account shared-device ring found with a permutation test. The repo contains the 20 final answers, graph loading steps, model evaluation, and runnable code. The challenge dataset stays out of the repo.” |

Keep the screen recording running through both live investigations, including
the wait for the VM. If the live URL is unavailable, restore the service and
restart the recording; a replay alone would not show the required end-to-end
agent run. The agent recommends bank actions but does not connect to bank card
controls or file reports. The Qwen fine-tune is an audited experiment, not the
deployed decision model.
