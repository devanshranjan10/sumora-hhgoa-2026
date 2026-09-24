import { useState } from 'react'
import { importPackage } from './api'

type Kind = 'case' | 'transaction'

const templates: Record<Kind, string> = {
  case: 'flagged_txn_id,trigger_type,trigger_text\nT3514030,analyst_request,Review unusual activity',
  transaction: 'transaction_id,customer_id,card_id,ts,amount_usd,channel,risk_score,device_profile,trigger_text\nT9000010,C90010,C90010-K1,2026-09-24 12:00:00,119.50,online,0.62,Example device,New transaction for review',
}

function parseCsv(input: string): Record<string, string>[] {
  const cells: string[][] = [[]]
  let quoted = false
  for (let index = 0; index < input.length; index += 1) {
    const character = input[index]
    const row = cells[cells.length - 1]
    if (character === '"') {
      if (quoted && input[index + 1] === '"') { row[row.length - 1] += '"'; index += 1 }
      else quoted = !quoted
    } else if (character === ',' && !quoted) row.push('')
    else if ((character === '\n' || character === '\r') && !quoted) {
      if (character === '\r' && input[index + 1] === '\n') index += 1
      cells.push([''])
    } else row[row.length - 1] += character
  }
  if (quoted) throw new Error('CSV has an unclosed quoted field')
  const rows = cells.filter(row => row.some(cell => cell.trim()))
  const headers = rows.shift()?.map(value => value.trim()) ?? []
  if (!headers.length) throw new Error('CSV needs a header row')
  if (headers.some(header => !header) || new Set(headers).size !== headers.length) throw new Error('CSV headers must be unique and non-empty')
  return rows.map((row, index) => {
    if (row.length !== headers.length) throw new Error(`Row ${index + 2} has ${row.length} fields; expected ${headers.length}`)
    return Object.fromEntries(headers.map((header, column) => [header, row[column].trim()]))
  })
}

export default function ImportView({ onImported }: { onImported: (caseIds: string[]) => void }) {
  const [kind, setKind] = useState<Kind>('case')
  const [fileName, setFileName] = useState('')
  const [rows, setRows] = useState<Record<string, string>[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [created, setCreated] = useState<string[]>([])

  async function readFile(file: File | undefined) {
    setError(''); setRows([]); setCreated([]); setFileName(file?.name ?? '')
    if (!file) return
    if (file.size > 64 * 1024) { setError('Use a CSV or JSON package smaller than 64 KB.'); return }
    try {
      const contents = await file.text()
      const parsed = file.name.toLowerCase().endsWith('.json') ? JSON.parse(contents) : parseCsv(contents)
      if (!Array.isArray(parsed) || !parsed.length || parsed.length > 5 || parsed.some(row => !row || typeof row !== 'object' || Array.isArray(row))) {
        throw new Error('A package must contain 1–5 object rows')
      }
      setRows(parsed as Record<string, string>[])
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not read package') }
  }

  async function submit() {
    if (!rows.length || busy) return
    setBusy(true); setError('')
    try {
      const ids = await importPackage(kind, rows)
      setCreated(ids)
      onImported(ids)
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Import failed') }
    finally { setBusy(false) }
  }

  return <div className="import-view">
    <div className="import-heading"><div><span className="eyebrow">ADD DATA</span><h2>Investigate your own package</h2><p>New cases are kept apart from the 20 scored submission files.</p></div></div>
    <div className="import-modes"><button className={kind === 'case' ? 'selected' : ''} onClick={() => { setKind('case'); setRows([]); setCreated([]); setFileName(''); setError('') }}>Existing graph transaction</button><button className={kind === 'transaction' ? 'selected' : ''} onClick={() => { setKind('transaction'); setRows([]); setCreated([]); setFileName(''); setError('') }}>New transaction row</button></div>
    <div className="import-layout">
      <section className="surface import-main">
        <h3>{kind === 'case' ? 'Create a case from loaded data' : 'Add transactions to TigerGraph'}</h3>
        <p>{kind === 'case' ? 'Reference a transaction already in the graph. The service resolves its card and customer directly from TigerGraph before creating a new case.' : 'Add a transaction with a customer, card, amount, channel, risk input, timestamp and device profile. The graph links are created before the agent runs. A new customer with no history receives a provisional assessment.'}</p>
        <label className="file-control"><span>{fileName || 'Choose CSV or JSON package'}</span><input type="file" accept=".csv,.json,text/csv,application/json" onChange={event => void readFile(event.target.files?.[0])} /></label>
        {rows.length > 0 && <div className="import-preview"><strong>{rows.length} row{rows.length === 1 ? '' : 's'} ready</strong><span>{Object.keys(rows[0]).join(' · ')}</span></div>}
        {error && <div className="import-error" role="alert">{error}</div>}
        {created.length > 0 && <div className="import-success" role="status">Added {created.join(', ')} to TigerGraph. Open the case and run the agent.</div>}
        <button className="import-submit" onClick={submit} disabled={!rows.length || busy || created.length > 0}>{busy ? 'Adding to graph…' : 'Add package to graph'}</button>
      </section>
      <section className="surface import-template"><span className="eyebrow">FILE FORMAT</span><h3>{kind === 'case' ? 'Case package' : 'Transaction package'}</h3><p>Header names must match. Upload one to five rows per package.</p><pre>{templates[kind]}</pre><p className="import-caveat">{kind === 'case' ? 'The transaction must exist in TigerGraph. Supplied card_id and customer_id, if present, are checked against graph edges.' : 'Use an unused T9 transaction ID. Rows on an existing card must follow its latest transaction. Sparse rows lack the full IEEE feature history; review the result before taking action.'}</p></section>
    </div>
  </div>
}
