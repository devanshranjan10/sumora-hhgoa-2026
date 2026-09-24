# Social post draft

I built Sumora for the TigerGraph × HHGoa fraud investigation task.

The part I care most about: when a case is genuinely unclear, the agent can ask
for step-up authentication and stop. It records the request as pending. It does
not fabricate a customer answer to make the case look resolved.

TigerGraph Community Edition holds the transaction and identity graph, earlier
cases, and the agent's audit trail. The live LangGraph run calls installed GSQL
queries through TigerGraph MCP, and the UI exposes the returned rows alongside
the recommendation. The 20 answer files, model evaluation, and runnable graph
setup are in the public repo: [insert public repo URL].

The hidden case labels are unavailable, so I am sharing the measured October
holdout result and limitations rather than claiming challenge accuracy.

@TigerGraphDB @247pmstudio #HHGoa #FraudDetection

Each team member should write and publish their own post, then paste every
public post URL into the form. Verify each URL opens without login.
