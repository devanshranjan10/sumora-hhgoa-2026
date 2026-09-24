import { useEffect, useRef, useState } from 'react'
import {
  loadAnswer, loadCaseIds, loadCatalog, loadCaseGraph, loadDiscovery, loadInvestigationRun,
  loadLatestCaseRun, loadLiveHealth, loadTrace, loadWatchlist, startInvestigation,
  type Answer, type CaseGraph, type Catalog, type DiscoveryArtifact, type LiveHealth, type Trace, type TraceStep, type Watchlist,
} from './api'
import GraphView from './GraphView'
import ImportView from './ImportView'
import WatchlistView from './WatchlistView'

type View = 'overview' | 'graph' | 'evidence' | 'activity' | 'discovery' | 'import' | 'watchlist'
type Action = Answer['next_best_actions']['final'][number]
const views: { id: View; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'graph', label: 'Graph' },
  { id: 'evidence', label: 'Evidence' },
  { id: 'activity', label: 'Agent activity' },
  { id: 'discovery', label: 'New typology' },
]
const title = (value: string) => value.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, letter => letter.toUpperCase())
const money = (value: number) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value)

function stopHuman(reason: string): { title: string; detail: string } {
  if (reason.startsWith('evsi_le_0')) return {
    title: 'More evidence would not change the decision',
    detail: 'The expected value of another request is zero or lower after its cost and delay.',
  }
  if (reason.startsWith('robust')) return {
    title: 'The decision holds across the confidence interval',
    detail: 'The policy choice remains cost-optimal throughout the estimated 90% confidence interval.',
  }
  if (reason.startsWith('p_extreme')) return {
    title: 'Independent signals agree',
    detail: 'The probability is outside the ambiguous band and two evidence classes support the route.',
  }
  return { title: title(reason), detail: '' }
}

function describeStep(step: TraceStep) {
  if (step.step === 'investigate') return `Queried ${((step.tools as string[]) ?? []).map(title).join(', ')}.`
  if (step.step === 'assess') return `Estimated p=${step.p} with 90% CI ${((step.ci as number[]) ?? []).join('–')}.`
  if (step.step === 'gate') return step.gather ? `Requested ${title(String(step.action))} because further evidence had value.` : `Selected ${title(String(step.action))}; ${String(step.stop ?? 'no further evidence required').replace(/_/g, ' ')}.`
  if (step.step === 'gather') return step.status === 'pending' ? `Requested ${title(String(step.type))}; response pending.` : `Recorded the supplied response for ${title(String(step.type))}.`
  if (step.step === 'approval') return `Approval route: ${String(step.route)}. ${step.approved ? 'Automatic' : 'Human approval required'}.`
  if (step.step === 'execute') return `Simulated recommendation: ${title(String(step.action))}.`
  if (step.step === 'update_memory') return 'Updated graph case memory.'
  if (step.step === 'explain') return 'Prepared evidence-backed explanation.'
  if (step.step === 'open_case') return 'Opened the case and its graph context.'
  if (step.step === 'close') return 'Completed the investigation.'
  return title(step.step)
}

function VerdictPill({ verdict }: { verdict?: string }) {
  return <span className={`verdict-pill verdict-${verdict ?? 'pending'}`}><span className="pill-dot" />{verdict ? title(verdict) : 'Pending'}</span>
}

function ActionList({ actions, empty }: { actions: Action[]; empty?: string }) {
  if (!actions.length) return <p className="empty-copy">{empty ?? 'No actions recorded.'}</p>
  return <div className="action-list">{actions.map((item, index) => (
    <div className="action-row" key={`${item.action}-${index}`}>
      <span className="action-index">{String(index + 1).padStart(2, '0')}</span>
      <div className="action-copy"><strong>{title(item.action)}</strong><span>{item.reason}</span></div>
      <span className="route-pill">{item.route}</span>
    </div>
  ))}</div>
}

function TraceTimeline({ trace, playing, stepCount }: { trace: Trace; playing: boolean; stepCount: number }) {
  return <div className="timeline">{trace.steps.map((step, index) => (
    <div key={`${step.step}-${index}`} className={`timeline-row ${playing && index >= stepCount ? 'waiting' : ''}`}>
      <span className="timeline-marker">{String(index + 1).padStart(2, '0')}</span>
      <div><strong>{title(step.step)}</strong><p>{describeStep(step)}</p></div>
    </div>
  ))}</div>
}

function McpEvidence({ trace }: { trace: Trace }) {
  const calls = trace.mcp_calls ?? []
  if (!calls.length) return null
  return <section className="mcp-evidence" aria-label="TigerGraph MCP query results">
    <div className="mcp-evidence-heading"><span className="eyebrow">TIGERGRAPH MCP · CAPTURED RESPONSE</span><h4>Installed query results</h4><p>Actual rows returned to the agent by the MCP tool during this run.</p></div>
    {calls.map((call, index) => <details key={`${call.query_name}-${index}`} open={index === 0}>
      <summary><code>{call.query_name}</code><span>{call.row_count} {call.row_count === 1 ? 'row' : 'rows'}</span></summary>
      <div className="mcp-evidence-body"><small>Tool</small><code>{call.tool}</code><small>Parameters</small><pre>{JSON.stringify(call.parameters, null, 2)}</pre><small>Result {call.truncated ? '(preview)' : ''}</small><pre>{call.result_preview}</pre></div>
    </details>)}
  </section>
}

function Discovery({ data }: { data: DiscoveryArtifact | null }) {
  if (!data) return <div className="empty-state">Discovery analysis is unavailable.</div>
  return <div className="discovery-view">
    <div className="discovery-hero">
      <div className="eyebrow light">Network discovery / research finding</div>
      <h2>{data.typology.name}</h2>
      <p>{data.typology.pattern_description}</p>
      <span className="research-badge">{data['significant_at_0.05'] && data.typology.shipped ? 'Statistically significant finding' : 'Research finding'}</span>
    </div>
    <div className="metrics-row">
      <div className="metric"><span>Observed rings</span><strong>{data.observed_ring_count}</strong></div>
      <div className="metric"><span>Null mean · 2,000 permutations</span><strong>{data.null_mean.toFixed(1)}</strong></div>
      <div className="metric"><span>Permutation p-value</span><strong>{data.perm_p_value}</strong></div>
    </div>
    <section className="surface discovery-detail">
      <div className="section-heading"><div><span className="eyebrow">Illustrative ring</span><h3>Shared device, separate cardholders</h3></div></div>
      <div className="ring-visual" aria-label={`${data.demo_exemplar.n_customers} cardholders and ${data.demo_exemplar.n_fraud_cases} cases share the exemplar device`}>
        <div className="ring-hub"><span>Shared device</span><strong>{data.demo_exemplar.n_customers}</strong><small>cardholders</small></div>
        <div className="ring-facts"><span>{data.demo_exemplar.n_fraud_cases} confirmed fraud cases</span><span>{data.demo_exemplar.case_ids.length} bank-seeded case links</span></div>
      </div>
      <p className="body-copy">The device fingerprint is the structural link. The bank-seeded exemplar spans {data.demo_exemplar.case_ids.join(', ')}.</p>
      <p className="technical-detail">Device profile: {data.demo_exemplar.device}</p>
    </section>
  </div>
}

export default function App() {
  const [ids, setIds] = useState<string[]>([])
  const [importedIds, setImportedIds] = useState<string[]>([])
  const [newTransactionIds, setNewTransactionIds] = useState<string[]>([])
  const [verdicts, setVerdicts] = useState<Record<string, string>>({})
  const [active, setActive] = useState('')
  const [query, setQuery] = useState('')
  const [view, setView] = useState<View>('overview')
  const [answer, setAnswer] = useState<Answer | null>(null)
  const [trace, setTrace] = useState<Trace | null>(null)
  const [recordedAnswer, setRecordedAnswer] = useState<Answer | null>(null)
  const [recordedTrace, setRecordedTrace] = useState<Trace | null>(null)
  const [discovery, setDiscovery] = useState<DiscoveryArtifact | null>(null)
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [watchlist, setWatchlist] = useState<Watchlist | null>(null)
  const [graph, setGraph] = useState<CaseGraph | null>(null)
  const [graphLoading, setGraphLoading] = useState(false)
  const [graphError, setGraphError] = useState('')
  const [health, setHealth] = useState<LiveHealth | null>(null)
  const [healthChecked, setHealthChecked] = useState(false)
  const [loadError, setLoadError] = useState('')
  const [liveError, setLiveError] = useState('')
  const [liveRunning, setLiveRunning] = useState(false)
  const [liveResult, setLiveResult] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [liveSteps, setLiveSteps] = useState<TraceStep[]>([])
  const [stepCount, setStepCount] = useState(0)
  const timer = useRef<number | null>(null)
  const activeRef = useRef(active)
  activeRef.current = active

  useEffect(() => {
    loadCaseIds().then((index) => {
      setIds(index.case_ids); setImportedIds(index.imported_case_ids); setNewTransactionIds(index.new_transaction_case_ids)
      if (index.case_ids.length) setActive(index.case_ids[0])
      return Promise.all(index.case_ids.map(async (id) => [id, index.imported_case_ids.includes(id)
        ? (await loadLatestCaseRun(id))?.answer.case.verdict ?? ''
        : (await loadAnswer(id))?.case.verdict ?? ''] as const))
    }).then((pairs) => setVerdicts(Object.fromEntries(pairs))).catch(() => setLoadError('Case files could not be loaded. Refresh the page to retry.'))
    loadDiscovery().then(setDiscovery).catch(() => setDiscovery(null))
    loadCatalog().then(setCatalog).catch(() => setCatalog(null))
    loadWatchlist().then(setWatchlist).catch(() => setWatchlist(null))
    loadLiveHealth().then(setHealth).catch(() => setHealth(null)).finally(() => setHealthChecked(true))
    return () => { if (timer.current) window.clearInterval(timer.current) }
  }, [])

  useEffect(() => {
    if (!active) return
    let current = true
    if (timer.current) window.clearInterval(timer.current)
    setAnswer(null); setTrace(null); setRecordedAnswer(null); setRecordedTrace(null); setStepCount(0); setPlaying(false)
    setLiveResult(false); setLiveError(''); setLoadError('')
    if (importedIds.includes(active)) {
      loadLatestCaseRun(active).then(result => {
        if (current && result) { setAnswer(result.answer); setTrace(result.trace); setLiveResult(true) }
      }).catch(() => { if (current) setLoadError('Could not load the latest investigation.') })
      return () => { current = false }
    }
    Promise.all([loadAnswer(active), loadTrace(active)]).then(([nextAnswer, nextTrace]) => {
      if (!current) return
      if (!nextAnswer) throw new Error('Answer file missing')
      setAnswer(nextAnswer); setTrace(nextTrace); setRecordedAnswer(nextAnswer); setRecordedTrace(nextTrace)
    }).catch(() => { if (current) setLoadError(`Could not load ${active}. Select another case or refresh.`) })
    return () => { current = false }
  }, [active, importedIds])

  useEffect(() => {
    if (!active || view !== 'graph') return
    let current = true
    setGraph(null); setGraphError(''); setGraphLoading(true)
    loadCaseGraph(active).then(value => { if (current) setGraph(value) })
      .catch(error => { if (current) setGraphError(error instanceof Error ? error.message : 'Graph unavailable') })
      .finally(() => { if (current) setGraphLoading(false) })
    return () => { current = false }
  }, [active, view])

  function selectCase(id: string) { setActive(id); setView('overview') }
  async function onImported(caseIds: string[]) {
    setIds(previous => [...new Set([...previous, ...caseIds])].sort())
    setImportedIds(previous => [...new Set([...previous, ...caseIds])])
    setActive(caseIds[0]); setView('graph')
    try {
      const index = await loadCaseIds()
      setIds(index.case_ids); setImportedIds(index.imported_case_ids); setNewTransactionIds(index.new_transaction_case_ids)
    } catch { setLoadError('Case was created, but the list could not refresh. Reload to see all imports.') }
  }
  function replay() {
    if (!trace || playing) return
    setView('activity'); setPlaying(true); setStepCount(0)
    if (timer.current) window.clearInterval(timer.current)
    let next = 0
    timer.current = window.setInterval(() => {
      next += 1; setStepCount(next)
      if (next >= trace.steps.length) {
        if (timer.current) window.clearInterval(timer.current)
        setPlaying(false)
      }
    }, 460)
  }
  async function runLive() {
    if (!active || liveRunning) return
    const requestedCase = active
    setLiveRunning(true); setLiveError(''); setLiveSteps([]); setView('activity')
    try {
      const runId = await startInvestigation(requestedCase)
      let result = await loadInvestigationRun(runId)
      while (result.status === 'running') {
        if (requestedCase === activeRef.current) setLiveSteps(result.steps)
        await new Promise(resolve => window.setTimeout(resolve, 500))
        result = await loadInvestigationRun(runId)
      }
      if (result.status === 'failed' || !result.answer || !result.trace) throw new Error(result.error ?? 'Investigation failed')
      const liveAnswer = result.answer
      const liveTrace = result.trace
      if (requestedCase === activeRef.current) {
        setAnswer(liveAnswer); setTrace(liveTrace)
        setVerdicts(previous => ({ ...previous, [requestedCase]: liveAnswer.case.verdict }))
        setLiveResult(true); setStepCount(liveTrace.steps.length); setPlaying(false)
        setLiveSteps(result.steps)
      }
      setHealth(await loadLiveHealth())
    } catch (error) {
      setLiveError(error instanceof Error ? error.message : 'Investigation failed')
    } finally { setLiveRunning(false) }
  }

  const ready = Boolean(health?.graph_ready && health.mcp_ready && health.model_ready)
  const awaitingInvestigation = importedIds.includes(active) && !answer
  const inCase = view !== 'import' && view !== 'watchlist'
  const visibleIds = ids.filter(id => id.toLowerCase().includes(query.trim().toLowerCase()))
  const probability = answer ? Math.round(answer.case.fraud_probability * 100) : null
  const stop = answer ? stopHuman(answer.stop_reason) : null
  const hasPattern = Boolean(answer?.case.pattern && !['none', 'unknown'].includes(answer.case.pattern.toLowerCase()))
  const casePattern = hasPattern && answer ? title(answer.case.pattern) : 'Pattern not assigned'
  const txnCount = answer?.case.affected_txn_ids.length ?? 0
  const priorCount = answer?.case.similar_prior_cases.length ?? 0
  const evidence = trace?.evidence ?? []
  const evidenceRequests = answer?.evidence_requests ?? []
  const changed = answer ? JSON.stringify(answer.next_best_actions.initial) !== JSON.stringify(answer.next_best_actions.final) : false
  const liveDiffers = Boolean(liveResult && answer && recordedAnswer && recordedTrace && (
    Math.abs(answer.case.fraud_probability - recordedAnswer.case.fraud_probability) > 0.05 ||
    answer.case.pattern !== recordedAnswer.case.pattern ||
    trace?.evidence[0]?.claim !== recordedTrace.evidence[0]?.claim
  ))

  return <div className="app-shell">
    <aside className="case-rail" aria-label="Case navigation">
      <div className="rail-brand"><span className="brand-mark" aria-hidden="true">S</span><div><strong>sumora</strong><span>Investigation OS</span></div></div>
      <div className="rail-caption">TIGERGRAPH <span>COMMUNITY EDITION</span></div>
      <button className={`rail-nav-active ${inCase ? 'selected' : ''}`} onClick={() => setView('overview')}>Submission cases <span className="nav-count">20</span></button>
      <button className={`rail-nav-import ${view === 'watchlist' ? 'selected' : ''}`} onClick={() => setView('watchlist')}>Watchlist <span>{watchlist?.matching_transactions ?? '•'}</span></button>
      <button className={`rail-nav-import ${view === 'import' ? 'selected' : ''}`} onClick={() => setView('import')}>Add data <span>＋</span></button>
      <div className="rail-section-head"><span>CASES</span><span>{ids.length} total</span></div>
      <label className="case-search"><span className="sr-only">Find a case</span><span aria-hidden="true">⌕</span><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Find a case" /></label>
      <nav className="case-list" aria-label="Cases">
        {importedIds.length > 0 && <span className="rail-group-label">IMPORTED · {importedIds.length}</span>}
        {visibleIds.filter(id => importedIds.includes(id)).map(id => <button type="button" key={id} className={`case-button ${id === active && inCase ? 'selected' : ''}`} onClick={() => selectCase(id)} aria-current={id === active && inCase ? 'page' : undefined}><span className="case-button-id">{id}</span><span className={`case-mini-dot dot-${verdicts[id] || 'pending'}`} aria-label={verdicts[id] || 'Pending'} /></button>)}
        <span className="rail-group-label">SUBMISSION PACK · 20</span>
        {visibleIds.filter(id => !importedIds.includes(id)).map(id => <button type="button" key={id} className={`case-button ${id === active && inCase ? 'selected' : ''}`} onClick={() => selectCase(id)} aria-current={id === active && inCase ? 'page' : undefined}>
          <span className="case-button-id">{id}</span><span className={`case-mini-dot dot-${verdicts[id] || 'pending'}`} aria-label={verdicts[id] || 'Pending'} />
        </button>)}
        {ids.length > 0 && visibleIds.length === 0 && <p className="rail-empty">No matching cases.</p>}
      </nav>
      <div className="rail-footer"><span className={`status-dot ${ready ? 'on' : ''}`} />{ready ? 'Graph + MCP + model online' : healthChecked ? 'Recorded results available' : 'Checking services…'}<small>{catalog ? `${catalog.transactions.toLocaleString()} transactions · ${catalog.closed_cases.toLocaleString()} closed cases` : 'Loading dataset size…'}</small></div>
    </aside>
    <div className="workspace">
      <header className="topbar"><div className="breadcrumbs"><span>Case pack</span><span className="crumb-slash">/</span><strong>{view === 'import' ? 'Add data' : view === 'watchlist' ? 'Watchlist' : active || 'Cases'}</strong></div><div className="topbar-right"><span className="topbar-environment"><span className={`status-dot ${ready ? 'on' : ''}`} />{ready ? 'TigerGraph connected' : 'Recorded mode'}</span><button className="topbar-link" onClick={() => setView(view === 'watchlist' ? 'overview' : 'watchlist')}>{view === 'watchlist' ? 'Return to case' : 'Watchlist'}</button><button className="topbar-link" onClick={() => setView(view === 'import' ? 'overview' : 'import')}>{view === 'import' ? 'Return to case' : 'Add data +'}</button></div></header>
      <main className="content">
        {loadError && <div className="error-banner" role="alert">{loadError}</div>}
        {inCase && <div className="page-intro"><div><span className="eyebrow">FRAUD INVESTIGATION</span><h1>{active || 'Case pack'}</h1><p>{answer ? `${casePattern} · ${answer.case.affected_txn_ids[0] ?? 'No flagged transaction'} · ${money(answer.case.exposure_usd)} exposure` : importedIds.includes(active) ? 'Imported case · ready for a live investigation' : 'Loading case…'}</p></div><div className="source-tag">{liveResult ? 'LIVE RESULT' : importedIds.includes(active) ? newTransactionIds.includes(active) ? 'NEW TRANSACTION' : 'IMPORTED CASE' : 'RECORDED ANSWER'}</div></div>}
        {inCase && <section className="decision-hero" aria-label="Case decision">
          <div className="hero-copy"><div className="hero-eyebrow">FRAUD ASSESSMENT <span className="hero-divider">/</span> {awaitingInvestigation ? 'NEW CASE' : casePattern.toUpperCase()}</div><div className="hero-verdict"><VerdictPill verdict={answer?.case.verdict} /><span className="hero-case-state">{answer ? title(answer.case.status) : awaitingInvestigation ? 'Ready' : 'Loading'}</span></div><h2>{answer ? (answer.case.verdict === 'fraud' ? 'Fraud risk detected' : answer.case.verdict === 'legitimate' ? 'Low fraud risk' : 'Further review required') : awaitingInvestigation ? 'Ready to investigate' : 'Loading case'}</h2><p>{answer ? `${txnCount} affected transaction${txnCount === 1 ? '' : 's'} · ${priorCount} related prior case${priorCount === 1 ? '' : 's'}` : awaitingInvestigation ? 'Run the agent to inspect the live graph and produce a recommendation.' : 'Loading graph evidence and decision record.'}</p><div className="hero-actions"><button type="button" className="primary-button" onClick={runLive} disabled={!ready || !active || liveRunning}>{liveRunning ? 'Agent working…' : 'Investigate live'} <span aria-hidden="true">→</span></button><button type="button" className="hero-secondary" onClick={replay} disabled={!trace || playing || liveRunning}>{playing ? 'Replaying…' : 'Replay recorded trace'}</button></div></div>
          <div className="hero-score"><span>FRAUD PROBABILITY</span><strong>{probability === null ? '—' : `${probability}%`}</strong><small>{probability === null ? 'Awaiting investigation' : liveResult ? 'Live model estimate' : 'Recorded estimate'}</small></div>
        </section>}
        {inCase && liveError && <div className="error-banner" role="alert">{liveError}</div>}
        {inCase && liveResult && <div className="success-banner" role="status"><span className="status-dot on" /> Live investigation completed against TigerGraph and the CPU model.</div>}
        {inCase && newTransactionIds.includes(active) && <div className="comparison-banner" role="note"><strong>Provisional assessment.</strong><span>This transaction was added after model training. Sparse customer history and a user-supplied risk score can make the estimate unreliable; review the graph evidence before acting.</span></div>}
        {inCase && liveDiffers && <div className="comparison-banner" role="note"><strong>Live and recorded results differ.</strong><span>Recorded: {Math.round(recordedAnswer!.case.fraud_probability * 100)}% · {title(recordedAnswer!.case.pattern)}. Live: {probability}% · {casePattern}. Review the cited evidence before using the recommendation.</span></div>}
        {inCase && <nav className="view-tabs" aria-label="Investigation sections">{views.map(item => <button type="button" key={item.id} className={view === item.id ? 'active' : ''} onClick={() => setView(item.id)} aria-current={view === item.id ? 'page' : undefined}>{item.label}{item.id === 'discovery' && <span className="tab-new">NEW</span>}</button>)}</nav>}

        {view === 'overview' && <div className="overview-grid">
          <section className="surface actions-surface"><div className="section-heading"><div><span className="eyebrow">Decision</span><h3>Recommended response</h3></div><span className="section-meta">FINAL ACTIONS</span></div><ActionList actions={answer?.next_best_actions.final ?? []} empty="Loading recommendations…" /><div className="surface-footer"><span className="footer-icon">i</span>{evidenceRequests.length ? `${evidenceRequests.length} evidence request${evidenceRequests.length === 1 ? '' : 's'} · recommendation ${changed ? 'changed' : 'confirmed'}` : 'No additional evidence was requested'}</div></section>
          <section className="surface rationale-surface"><div className="section-heading"><div><span className="eyebrow">Decision logic</span><h3>Why the agent stopped</h3></div></div><div className="rationale-symbol">✓</div><h4>{stop?.title ?? 'Loading decision logic…'}</h4><p>{stop?.detail ?? ''}</p><span className="rationale-code">{answer?.tool_calls ?? '—'} tool calls <span>·</span> {answer?.tokens ?? '—'} tokens</span></section>
          <section className="surface evidence-surface"><div className="section-heading"><div><span className="eyebrow">Graph context</span><h3>Evidence at a glance</h3></div><button className="text-button" onClick={() => setView('evidence')}>View all evidence <span aria-hidden="true">↗</span></button></div><div className="evidence-preview">{evidence.filter(item => item.source === 'graph').slice(0, 2).map((item, index) => <div className="evidence-preview-row" key={`${item.ref}-${index}`}><span className="evidence-icon">{index + 1}</span><div><strong>{item.claim}</strong><small>{item.ref}</small></div></div>)}{!evidence.length && <p className="empty-copy">Loading evidence…</p>}</div></section>
          <section className="surface facts-surface"><div className="section-heading"><div><span className="eyebrow">Case profile</span><h3>At a glance</h3></div></div><div className="fact-row"><span>Exposure</span><strong>{answer ? money(answer.case.exposure_usd) : '—'}</strong></div><div className="fact-row"><span>Transactions</span><strong>{txnCount}</strong></div><div className="fact-row"><span>Prior cases cited</span><strong>{priorCount}</strong></div><div className="fact-row"><span>SAR recommendation</span><strong>{answer ? answer.sar.file ? 'Prepare report' : 'No report' : '—'}</strong></div></section>
          <section className="surface sar-surface"><div className="section-heading"><div><span className="eyebrow">Reporting</span><h3>Suspicious activity report</h3></div><span className={`report-pill ${answer?.sar.file ? 'report-yes' : ''}`}>{answer?.sar.file ? 'PREPARE' : 'NOT RECOMMENDED'}</span></div><p>{answer?.sar.reason ?? 'Loading reporting recommendation…'}</p>{answer?.sar.file && <><p className="body-copy">{answer.sar.narrative}</p><div className="sar-total">{money(answer.sar.total_amount_usd)} <span>across {answer.sar.subjects.length} subject{answer.sar.subjects.length === 1 ? '' : 's'}</span></div></>}</section>
        </div>}

        {view === 'graph' && <GraphView graph={graph} loading={graphLoading} error={graphError} />}

        {view === 'evidence' && <div className="detail-grid"><section className="surface detail-main"><div className="section-heading"><div><span className="eyebrow">Graph + policy</span><h3>Cited evidence</h3></div><span className="section-meta">{evidence.length} ITEMS</span></div>{evidence.map((item, index) => <div className="evidence-detail-row" key={`${item.ref}-${index}`}><span className="evidence-number">{String(index + 1).padStart(2, '0')}</span><div><span className="source-pill">{item.source}</span><p>{item.claim}</p><small>{item.ref}</small></div></div>)}{!evidence.length && <p className="empty-copy">Evidence is loading or unavailable.</p>}</section><div className="detail-side"><section className="surface"><div className="section-heading"><div><span className="eyebrow">Linked records</span><h3>Graph connections</h3></div></div><div className="fact-row"><span>Prior cases</span><strong>{priorCount}</strong></div><div className="record-chips">{answer?.case.similar_prior_cases.map(id => <span key={id}>{id}</span>)}</div><div className="fact-row"><span>Connected cards</span><strong>{answer?.case.connected_card_ids.length ?? 0}</strong></div><div className="record-chips">{answer?.case.connected_card_ids.map(id => <span key={id}>{id}</span>)}</div><div className="fact-row"><span>Transactions</span><strong>{txnCount}</strong></div><div className="record-chips">{answer?.case.affected_txn_ids.map(id => <span key={id}>{id}</span>)}</div></section><section className="surface"><div className="section-heading"><div><span className="eyebrow">Active learning</span><h3>Evidence requests</h3></div></div>{evidenceRequests.length ? evidenceRequests.map((request, index) => <div className="request-row" key={`${request.type}-${index}`}><strong>{title(request.type)}</strong><p>Response status: {request.assumed_response}</p></div>) : <p className="empty-copy">No further evidence was requested for this case.</p>}</section></div></div>}

        {view === 'activity' && <div className="detail-grid"><section className="surface detail-main"><div className="section-heading"><div><span className="eyebrow">{liveRunning ? 'LIVE / RUNNING' : liveResult ? 'LIVE / COMPLETE' : 'RECORDED RUN'}</span><h3>Agent activity</h3></div><span className="section-meta">{liveRunning ? `${liveSteps.length} STEPS COMPLETE` : `${trace?.steps.length ?? 0} STEPS · ${trace?.tool_log.length ?? 0} TOOLS`}</span></div>{liveRunning ? <div className="live-step-list">{liveSteps.map((step, index) => <div className="live-step" key={`${step.step}-${index}`}><span>{String(index + 1).padStart(2, '0')}</span><div><strong>{title(step.step)}</strong><p>{describeStep(step)}</p></div><b>DONE</b></div>)}<div className="live-step current"><span>{String(liveSteps.length + 1).padStart(2, '0')}</span><div><strong>Working in TigerGraph</strong><p>Waiting for the next actual agent step.</p></div><b>LIVE</b></div></div> : trace ? <><TraceTimeline trace={trace} playing={playing} stepCount={stepCount} /><McpEvidence trace={trace} /></> : <p className="empty-copy">Trace is loading or unavailable.</p>}</section><div className="detail-side"><section className="surface"><div className="section-heading"><div><span className="eyebrow">Before evidence</span><h3>Initial recommendation</h3></div></div><ActionList actions={answer?.next_best_actions.initial ?? []} /></section><section className="surface"><div className="section-heading"><div><span className="eyebrow">After investigation</span><h3>Current recommendation</h3></div></div><ActionList actions={answer?.next_best_actions.final ?? []} /><div className="surface-footer">{answer?.next_best_actions.what_changed && answer.next_best_actions.what_changed !== 'nothing' ? answer.next_best_actions.what_changed : 'Recommendation held after investigation.'}</div></section></div></div>}
        {view === 'discovery' && <Discovery data={discovery} />}
        {view === 'import' && <ImportView onImported={ids => void onImported(ids)} />}
        {view === 'watchlist' && <WatchlistView data={watchlist} onImported={ids => void onImported(ids)} />}
        <footer className="content-footer"><span>SUMORA / INVESTIGATION CONSOLE</span><span>Recommendations are simulated; human approval routes remain in force.</span></footer>
      </main>
    </div>
  </div>
}
