import type { GridState } from './api'

export type Campus = {
  bus: number
  capacity_mw: number
  processing_mw: number
  soc: number
  backlog_mwh: number
  battery_enabled: boolean
}

export type Violation = {
  kind: string
  equipment_id: number | string | null
  timestep: number
  value: number | null
  limit: number | null
  severity: number | string
}

export type Region = 'benchmark' | 'singapore' | 'virginia'

export type MissionGrid = Omit<GridState, 'model' | 'action'> & {
  campuses: Campus[]
  violations: Violation[]
  model_version: string | null
  region?: Region
  region_details?: {name: string; peak_mw: number; reference: string; source: string} | null
  seed: number
  mode?: string
  rejected?: boolean
  clipped?: boolean
  eligible_buses?: number[]
  disabled_batteries?: number[]
}

export type Frame = {
  tick: number
  loading: Array<number | null>
  voltage: Array<number | null>
  campuses: Campus[]
  violations: Violation[]
}

export type Allocation = {bus: number; mw: number}
export type Proposal = {
  id: string | number
  allocation: Allocation[]
  accepted_mw: number
  source: string
  verified: boolean
  passed: boolean
  conditional?: boolean
  witness?: Violation | null
  trajectory: Frame[]
  imagined?: Frame[] | null
  metrics?: Record<string, unknown>
  case?: string
  policy?: Dispatch
  parent_id?: Proposal['id'] | null
  grade?: {failed_cases: number; violations: number; completed_mwh: number; deadline_miss_mwh: number}
  cases?: Array<{id: string; passed: boolean; witness?: Violation | null}>
}

export type Study = {
  id: string
  revision: number
  status: string
  progress: string
  proposals: Proposal[]
  best?: Proposal | Proposal['id'] | null
  solves: number
  activity?: unknown
  scope?: string
  model_version?: string
  inference_backend?: string
  tested_cases?: string[]
  baseline?: Record<string, Violation | null>
  islanding?: Record<string, number[]>
}

export type MissionEdit = {
  kind: 'compute' | 'demand' | 'rating' | 'line' | 'eligible' | 'battery' | 'bus'
  value: number | boolean | number[]
  target?: number
  hidden?: boolean
}

export type Dispatch = 'hold' | 'heuristic' | 'learned' | 'assisted' | 'flex'
export type Pick = {type: 'line' | 'bus'; id: number}

export type RunStatus = {
  model_version: string
  available: boolean
  inference_backend?: string
  training: Array<{step?: number; stage?: string}>
  evaluation: unknown
}
