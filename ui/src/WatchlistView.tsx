import { useState } from 'react'
import { importPackage, type Watchlist } from './api'

const money = (value: number) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value)

export default function WatchlistView({ data, onImported }: { data: Watchlist | null; onImported: (ids: string[]) => void }) {
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')

  async function openCase(transactionId: string) {
    if (busy) return
    setBusy(transactionId); setError('')
    try {
      const ids = await importPackage('case', [{
        flagged_txn_id: transactionId,
        trigger_type: 'analyst_request',
        trigger_text: 'Autonomous watchlist: low bank risk with earlier confirmed fraud on a shared device profile',
      }])
      onImported(ids)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not create the case')
    } finally { setBusy('') }
  }

  return <div className="watchlist-view">
    <div className="watchlist-header"><span className="eyebrow">AUTONOMOUS MONITORING / EXAM PERIOD</span><h2>Low score. Shared fraud history.</h2><p>Unsubmitted November–December transactions with a bank risk score below 0.50, linked by a Build-qualified device profile to earlier confirmed fraud on other customers.</p></div>
    {error && <div className="error-banner" role="alert">{error}</div>}
    {!data ? <div className="plain-state">Scanning the exam period…</div> : <>
      <div className="watchlist-metrics"><div><span>Matching transactions</span><strong>{data.matching_transactions.toLocaleString()}</strong></div><div><span>Shown for review</span><strong>{data.items.length}</strong></div><div><span>Bank risk ceiling</span><strong>&lt; 0.50</strong></div></div>
      <p className="watchlist-method">{data.method} Device profiles are coarse and can collide across people; each alert requires graph evidence and human review.</p>
      <div className="watchlist-rows">{data.items.map((item, index) => <article className="watchlist-row" key={item.transaction_id}>
        <div className="watchlist-rank">{String(index + 1).padStart(2, '0')}</div>
        <div className="watchlist-main"><strong>{item.transaction_id}</strong><span>{item.customer_id} · {item.timestamp.slice(0, 10)} · {money(item.amount_usd)}</span><small>{item.device_profile}</small></div>
        <div className="watchlist-evidence"><strong>{item.prior_customer_count}</strong><span>prior customers</span><small>{item.prior_case_count} confirmed cases · risk {item.bank_risk_score.toFixed(2)}</small></div>
        <button type="button" disabled={Boolean(busy)} onClick={() => void openCase(item.transaction_id)}>{busy === item.transaction_id ? 'Opening…' : 'Open case →'}</button>
      </article>)}</div>
    </>}
  </div>
}
