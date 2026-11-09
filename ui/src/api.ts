export type Action = {kind: 'hold' | 'cap' | 'dispatch' | 'line'; a?: number; b?: number; value?: number}
export type Mode = 'manual' | 'suggest' | 'auto' | 'assisted'
export type PlantPreset = 'senoko' | 'surry' | 'drax'
export type Scenario = 'baseline' | 'outage' | 'hidden' | PlantPreset
export type Bus = {id: number; voltage: number | null; demand: number; generation: number; eligible?: boolean; name?: string | null; position?: [number, number] | null}
export type Line = {id: number; from: number; to: number; loading: number | null; flow: number | null; active: boolean; rating: number}
export type GridState = {
  id: string
  revision: number
  tick: number
  buses: Bus[]
  lines: Line[]
  compute: {bus: number; requested: number; cap: number; served: number}
  events: unknown[]
  model: unknown
  action?: Action | null
  network?: string
  hidden?: boolean
  solver_intervened?: boolean
  settings?: {demand_scale: number; hidden_bus: number; hidden_scale: number; rating_scale: number[]}
  preset?: PlantPreset | null
  site?: {name: string; region: string; technology: string; anchor_bus: number; grid: 'synthetic'} | null
}
export type Edit = {kind: 'compute' | 'demand' | 'rating' | 'line' | 'bus'; value: number | boolean; target?: number; hidden?: boolean}
export type Prediction = {loading: Array<number | null>; voltage: Array<number | null>; served: number; existing: number; event: unknown}
export type Candidate = {action: Action; plan?: Action[]; trajectory: Prediction[]; score: unknown; source?: string}
export type Forecast = {revision: number; candidates: Candidate[]; model: unknown; latency_ms?: number}
export type CapacityCase = {outage: number | null; feasible: boolean; failure: boolean; worst_line: number; loading: number; lost: number; min_voltage: number}
export type CapacityProposal = {bus: number; cap_mw: number; feasible: boolean; cases: CapacityCase[]; source?: 'wam' | 'search'; predicted_violations?: number}
export type CapacityStudy = {revision: number; requested_mw: number; tested_outages: number[]; baseline_infeasible_outages: number[]; attempts: CapacityProposal[]; best: CapacityProposal | null; scope: string}

export async function api<T>(path: string, body?: object, signal?: AbortSignal, base = '/api'): Promise<T> {
  const response = await fetch(base + path, body === undefined ? {signal} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body), signal,
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(typeof payload.detail === 'string' ? payload.detail : `API returned ${response.status}`)
  }
  return response.json()
}

export function actionName(action?: Action | null) {
  switch (action?.kind) {
    case 'cap': return `Set compute cap · ${Math.round((action.value ?? 0) * 100)}%`
    case 'dispatch': return `Redispatch · ${action.a} → ${action.b}`
    case 'line': return `${action.value ? 'Close' : 'Open'} line ${action.a}`
    case 'hold': return 'Hold current settings'
    default: return 'Awaiting decision'
  }
}
