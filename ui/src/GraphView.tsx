import { useMemo, useState } from 'react'
import type { CaseGraph } from './api'

type GraphNode = CaseGraph['nodes'][number]

function positionNodes(nodes: GraphNode[]) {
  const transactions = nodes.filter(node => node.type === 'Transaction')
  const entities = nodes.filter(node => node.type !== 'Transaction' && node.type !== 'FraudCase')
  const positions = new Map<string, { x: number; y: number }>()
  nodes.filter(node => node.type === 'FraudCase').forEach(node => positions.set(node.id, { x: 100, y: 82 }))
  transactions.sort((a, b) => Number(b.focus) - Number(a.focus) || Number(b.details.bank_risk_score ?? 0) - Number(a.details.bank_risk_score ?? 0))
  transactions.forEach((node, index) => positions.set(node.id, { x: 405, y: 82 + index * 56 }))
  entities.forEach((node, index) => positions.set(node.id, { x: 735, y: 82 + index * 72 }))
  return { positions, height: Math.max(400, 160 + Math.max(transactions.length * 56, entities.length * 72)) }
}

function nodeCaption(node: GraphNode) {
  if (node.type === 'DeviceProfile') return 'Device profile'
  if (node.type === 'Card') return 'Card'
  if (node.type === 'EmailAddress') return 'Email'
  if (node.type === 'BillingRegion') return 'Region'
  if (node.type === 'IdentityCluster') return 'Identity cluster'
  if (node.type === 'FraudCase') return 'Case'
  return node.focus ? 'Flagged transaction' : 'Related transaction'
}

export default function GraphView({ graph, loading, error }: { graph: CaseGraph | null; loading: boolean; error: string }) {
  const [selectedId, setSelectedId] = useState('')
  const selected = graph?.nodes.find(node => node.id === selectedId) ?? graph?.nodes.find(node => node.focus) ?? null
  const layout = useMemo(() => positionNodes(graph?.nodes ?? []), [graph])

  if (loading) return <div className="plain-state">Reading the case neighborhood from TigerGraph…</div>
  if (error) return <div className="plain-state error" role="alert">{error}</div>
  if (!graph) return <div className="plain-state">No graph neighborhood is available.</div>

  return <div className="graph-workspace">
    <div className="graph-topline"><div><strong>Live graph neighborhood</strong><span>{graph.nodes.length} visible nodes · {graph.edges.length} visible links</span></div><div><span>{graph.txn_candidates} transaction candidates</span><span>{graph.edge_candidates} edge candidates</span></div><span className="graph-swipe-hint">Swipe graph →</span></div>
    <div className="graph-scroll">
      <svg className="graph-canvas" viewBox={`0 0 940 ${layout.height}`} role="img" aria-label={`TigerGraph neighborhood for ${graph.case_id}`}>
        <text x="64" y="32" className="graph-column-label">CASE</text><text x="350" y="32" className="graph-column-label">TRANSACTIONS</text><text x="680" y="32" className="graph-column-label">CONNECTED ENTITIES</text>
        {graph.edges.map((edge, index) => {
          const source = layout.positions.get(edge.source)
          const target = layout.positions.get(edge.target)
          if (!source || !target) return null
          return <line key={`${edge.source}-${edge.target}-${index}`} x1={source.x + 78} y1={source.y + 20} x2={target.x - 78} y2={target.y + 20} className={edge.type === 'INVESTIGATES' ? 'graph-edge focus-edge' : 'graph-edge'} />
        })}
        {graph.nodes.map(node => {
          const point = layout.positions.get(node.id)
          if (!point) return null
          const chosen = selected?.id === node.id
          return <g key={node.id} role="button" tabIndex={0} aria-label={`${nodeCaption(node)} ${node.label}`} onClick={() => setSelectedId(node.id)} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') setSelectedId(node.id) }} className="graph-node" transform={`translate(${point.x - 78},${point.y})`}>
            <rect width="156" height="40" rx="5" className={`graph-node-box ${chosen ? 'chosen' : ''} ${node.focus ? 'focus' : ''}`} />
            <text x="12" y="16" className={`graph-node-type ${chosen || node.focus ? 'inverted' : ''}`}>{nodeCaption(node).toUpperCase()}</text>
            <text x="12" y="31" className={`graph-node-name ${chosen || node.focus ? 'inverted' : ''}`}>{node.label.length > 19 ? `${node.label.slice(0, 17)}…` : node.label}</text>
          </g>
        })}
      </svg>
    </div>
    {selected && <div className="graph-inspector"><div><span className="eyebrow">SELECTED NODE</span><strong>{selected.label}</strong><span>{nodeCaption(selected)}</span></div><dl>{Object.entries(selected.details).map(([key, value]) => <div key={key}><dt>{key.replace(/_/g, ' ')}</dt><dd>{value}</dd></div>)}</dl></div>}
    <p className="graph-footnote">A bounded view of the live TigerGraph neighborhood. Select a node to inspect its returned attributes; candidate counts include records outside this view.</p>
  </div>
}
