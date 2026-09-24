export interface TraceStep {
  step: string
  [k: string]: unknown
}

export interface Trace {
  case_id: string
  steps: TraceStep[]
  tool_log: [string, number][]
  mcp_calls?: {
    tool: string
    query_name: string
    parameters: Record<string, unknown>
    row_count: number
    result_preview: string
    truncated: boolean
  }[]
  p_final: number
  p_initial: number
  evsi_table: Record<string, number>
  stop_reason: string
  evidence: { claim: string; source: string; ref: string; entity_ids: string[] }[]
  requests: { type: string; asked_after_step: number; assumed_response: string }[]
  similar_prior_cases: string[]
}

export interface Answer {
  case_id: string
  case: {
    status: string
    verdict: string
    fraud_probability: number
    pattern: string
    pattern_description?: string
    affected_txn_ids: string[]
    connected_card_ids: string[]
    exposure_usd: number
    similar_prior_cases: string[]
    summary: string
  }
  evidence_requests: { type: string; asked_after_step: number; assumed_response: string }[]
  next_best_actions: {
    initial: { action: string; route: string; reason: string }[]
    final: { action: string; route: string; reason: string }[]
    what_changed: string
  }
  sar: {
    file: boolean
    reason: string
    narrative: string
    subjects: string[]
    total_amount_usd: number
    activity_dates: string[]
  }
  stop_reason: string
  tool_calls: number
  tokens: number
}

export interface DiscoveryArtifact {
  'significant_at_0.05': boolean
  perm_p_value: number
  observed_ring_count: number
  null_mean: number
  typology: {
    pattern: string
    pattern_id: string
    name: string
    pattern_description: string
    shipped: boolean
  }
  demo_exemplar: {
    device: string
    n_customers: number
    n_fraud_cases: number
    case_ids: string[]
    customer_ids: string[]
  }
}

export interface LiveHealth {
  graph_ready: boolean
  mcp_ready: boolean
  model_ready: boolean
  case_count: number
  deployment: string
}

export interface Catalog {
  submission_cases: number
  imported_cases: number
  transactions: number
  closed_cases: number
  customers: number
  cards: number
  device_profiles: number
}

export interface Watchlist {
  method: string
  matching_transactions: number
  items: {
    transaction_id: string
    customer_id: string
    timestamp: string
    amount_usd: number
    bank_risk_score: number
    prior_confirmed_cases: string[]
    prior_case_count: number
    prior_customer_count: number
    device_customer_count: number
    device_profile: string
  }[]
}

export async function loadWatchlist(): Promise<Watchlist> {
  const r = await fetch('/api/watchlist')
  if (!r.ok) throw new Error('Autonomous watchlist is unavailable')
  return r.json()
}

export interface CaseGraph {
  case_id: string
  nodes: { id: string; label: string; type: string; focus: boolean; details: Record<string, string> }[]
  edges: { source: string; target: string; type: string }[]
  txn_candidates: number
  edge_candidates: number
}

export interface InvestigationRun {
  case_id: string
  status: 'running' | 'complete' | 'failed'
  steps: TraceStep[]
  answer?: Answer
  trace?: Trace
  error?: string
}

export interface CaseIndex {
  case_ids: string[]
  imported_case_ids: string[]
  new_transaction_case_ids: string[]
}

export async function loadCatalog(): Promise<Catalog> {
  const r = await fetch('/api/catalog')
  if (!r.ok) throw new Error('Dataset summary is unavailable')
  return r.json()
}

export async function loadCaseGraph(id: string): Promise<CaseGraph> {
  const r = await fetch(`/api/cases/${encodeURIComponent(id)}/graph`)
  if (!r.ok) throw new Error('Graph neighborhood is unavailable')
  return r.json()
}

export async function loadLatestCaseRun(id: string): Promise<{ answer: Answer; trace: Trace } | null> {
  const r = await fetch(`/api/cases/${encodeURIComponent(id)}/latest`)
  if (r.status === 404) return null
  if (!r.ok) throw new Error('Latest investigation is unavailable')
  return r.json()
}

export async function startInvestigation(id: string): Promise<string> {
  const r = await fetch(`/api/cases/${encodeURIComponent(id)}/runs`, { method: 'POST' })
  if (!r.ok) {
    const body = await r.json().catch(() => null) as { detail?: string } | null
    throw new Error(body?.detail ?? `Investigation failed (${r.status})`)
  }
  return (await r.json() as { run_id: string }).run_id
}

export async function loadInvestigationRun(runId: string): Promise<InvestigationRun> {
  const r = await fetch(`/api/runs/${encodeURIComponent(runId)}`)
  if (!r.ok) throw new Error('Could not read the live run')
  return r.json()
}

export async function loadLiveHealth(): Promise<LiveHealth> {
  const r = await fetch('/api/health')
  if (!r.ok) throw new Error('Live service is unavailable')
  return r.json()
}

export async function runLiveInvestigation(id: string): Promise<{ answer: Answer; trace: Trace }> {
  const r = await fetch(`/api/cases/${encodeURIComponent(id)}/investigate`, { method: 'POST' })
  if (!r.ok) {
    const body = await r.json().catch(() => null) as { detail?: string } | null
    throw new Error(body?.detail ?? `Investigation failed (${r.status})`)
  }
  return r.json()
}

export async function loadCaseIds(): Promise<CaseIndex> {
  const r = await fetch('/api/cases')
  if (!r.ok) throw new Error('Case list is unavailable')
  return r.json()
}

export async function importPackage(kind: 'case' | 'transaction', rows: Record<string, string>[]): Promise<string[]> {
  const r = await fetch('/api/import', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ kind, rows }),
  })
  if (!r.ok) {
    const body = await r.json().catch(() => null) as { detail?: string } | null
    throw new Error(body?.detail ?? `Import failed (${r.status})`)
  }
  return (await r.json() as { case_ids: string[] }).case_ids
}

export async function loadTrace(id: string): Promise<Trace> {
  const r = await fetch(`/data/traces/${id}.trace.json`)
  if (!r.ok) throw new Error(`trace ${id} missing`)
  return r.json()
}

export async function loadAnswer(id: string): Promise<Answer | null> {
  const r = await fetch(`/data/answers/${id}.json`)
  if (!r.ok) return null
  return r.json()
}

export async function loadDiscovery(): Promise<DiscoveryArtifact | null> {
  const r = await fetch('/data/discovery.json')
  if (!r.ok) return null
  return r.json()
}
