import { Component, Suspense, useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { api, actionName, type Candidate, type CapacityCase, type CapacityStudy, type Edit, type Forecast, type GridState, type Mode, type Scenario } from './api'
import { GridScene, lineStatus } from './scene'
import { Mission } from './Mission'

type Pick = {type: 'bus' | 'line'; id: number}
const mw = (number?: number | null) => number == null ? '—' : `${number.toFixed(1)} MW`
const percent = (number?: number | null) => number == null ? '—' : `${Math.round(number * 100)}%`
const finite = (values: Array<number | null>) => values.filter((v): v is number => v != null && Number.isFinite(v))
const peak = (values: Array<number | null>) => {const xs = finite(values); return xs.length ? Math.max(...xs) : null}
const low = (values: Array<number | null>) => {const xs = finite(values); return xs.length ? Math.min(...xs) : null}
const clock = (tick: number) => `T+${String(Math.floor(tick * 5 / 60)).padStart(2,'0')}:${String(tick * 5 % 60).padStart(2,'0')}`
const metrics = {
  loading: {label:'Peak line loading', unit:percent, limit:1, bottom:0},
  voltage: {label:'Lowest bus voltage', unit:(value: number) => `${value.toFixed(2)} p.u.`, limit:.9, bottom:.8},
  served: {label:'Compute served', unit:mw, limit:null, bottom:0},
}
type Metric = keyof typeof metrics

class SceneBoundary extends Component<{children: ReactNode}, {error: boolean}> {
  state = {error:false}
  static getDerivedStateFromError() { return {error:true} }
  render() { return this.state.error ? <div className="empty"><h2>3D view unavailable</h2><p>The grid inspector and simulator controls remain available.</p></div> : this.props.children }
}

function Range({ label, value, min, max, step, suffix, disabled, onPreview, onCommit, onCancel }: {
  label: string; value: number; min: number; max: number; step: number; suffix: string; disabled: boolean
  onPreview: (n: number) => void; onCommit: (n: number) => void; onCancel: () => void
}) {
  const [draft, setDraft] = useState<number | null>(null)
  const pending = useRef<number | null>(null)
  const commit = () => {
    const next = pending.current
    if (next === null) return
    pending.current = null
    setDraft(null)
    if (next !== value) onCommit(next)
    else onCancel()
  }
  return <label className="range">
    <span>{label}<strong>{(draft ?? value).toLocaleString(undefined, {maximumFractionDigits: 2})}{suffix}</strong></span>
    <input type="range" aria-label={label} value={draft ?? value} min={min} max={max} step={step} disabled={disabled}
      onChange={event => {
        const next = Number(event.currentTarget.value)
        pending.current = next
        setDraft(next)
        onPreview(next)
      }}
      onPointerDown={event => event.currentTarget.setPointerCapture(event.pointerId)}
      onPointerUp={commit} onPointerCancel={commit} onKeyUp={commit} onBlur={commit}/>
    <small>{min}{suffix}<i/>{max}{suffix}</small>
  </label>
}

function Plot({ forecast, actual, hold, metric }: {
  forecast: Array<number | null>; actual: Array<number | null>; hold: Array<number | null>; metric: Metric
}) {
  const {label, unit, limit, bottom} = metrics[metric]
  const values = finite([...forecast, ...actual, ...hold])
  if (!values.length) return <div className="plot-placeholder">Run a forecast to see the next 30 minutes.</div>
  const floor = Math.min(bottom, ...values)
  const ceiling = Math.max((limit ?? 0) * 1.04, floor + .05, ...values) * 1.02
  const x = (index: number) => 53 + index * 35
  const y = (value: number) => 91 - (value - floor) / (ceiling - floor) * 68
  const d = (series: Array<number | null>) => series.map((v, i) => v == null ? '' : `${i === 0 || series[i - 1] == null ? 'M' : 'L'} ${x(i)} ${y(v)}`).join(' ')
  return <figure className="plot">
    <svg viewBox="0 0 252 112" role="img" aria-label={`${label} for forecast and observed outcomes over time`}>
      {limit !== null && <><path d={`M53 ${y(limit)} H231`} stroke="#d5cfc0" strokeDasharray="2 4"/><text x="0" y={y(limit)+3}>{unit(limit)}</text></>}
      <path d="M53 91 H231" stroke="#d5cfc0"/>
      <text x="0" y="94">{unit(floor)}</text>
      <path d={d(hold)} fill="none" stroke="#787c71" strokeDasharray="2 4" strokeWidth="1.7"/>
      <path d={d(forecast)} fill="none" stroke="#c04934" strokeDasharray="5 4" strokeWidth="2"/>
      <path d={d(actual)} fill="none" stroke="#26746a" strokeWidth="2"/>
      {actual.map((v, i) => v == null ? null : <circle key={i} cx={x(i)} cy={y(v)} r="2.6" fill="#26746a"/>)}
      <text x="53" y="108">+5m</text><text x="203" y="108">+30m</text>
    </svg>
    <figcaption><span className="key forecast-key">Forecast</span><span className="key actual-key">Observed</span>{hold.length > 0 && <span className="key hold-key">Hold branch</span>}</figcaption>
  </figure>
}

function describeEvent(value: unknown): string {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.map(describeEvent).filter(Boolean).join(' · ')
  if (value && typeof value === 'object') {
    const item = value as Record<string, unknown>
    return String(item.message ?? item.kind ?? item.type ?? item.reason ?? JSON.stringify(item))
  }
  return value ? String(value) : ''
}

function modelLabel(model: unknown) {
  if (typeof model === 'string') return model.includes('smoke') ? 'Local smoke checkpoint · demo' : model
  if (model && typeof model === 'object') {
    const value = model as Record<string, unknown>
    return String(value.run_id ?? value.checkpoint ?? value.name ?? value.status ?? 'Model connected')
  }
  return 'Simulator connected'
}

function capacityConstraint(outcome: CapacityCase) {
  if (outcome.failure) return 'AC solve failed'
  if (outcome.lost > .001) return `${mw(outcome.lost)} of existing load disconnected`
  if (outcome.loading > 1) return `Line ${outcome.worst_line} reached ${percent(outcome.loading)} of its rating`
  if (outcome.min_voltage < .9) return `Lowest bus voltage fell to ${outcome.min_voltage.toFixed(3)} p.u.`
  return 'Operating limit violation'
}

function CapacityResult({ study, stale }: {study: CapacityStudy; stale: boolean}) {
  const proposals = study.attempts.filter(attempt => attempt.source === 'wam')
  const rejected = proposals.find(proposal => !proposal.feasible) ??
    study.attempts.find(proposal => proposal.cap_mw > 0 && !proposal.feasible)
  const failed = rejected?.cases.find(outcome => !outcome.feasible)
  const context = failed?.outage != null ? ` with line ${failed.outage} open` : ''
  return <div className="capacity-result">
    {proposals.length > 0 && <div className="capacity-proposals">
      <b>WAM proposals</b>
      {proposals.map(proposal => <div key={proposal.bus}>
        <span>Bus {proposal.bus} · {mw(proposal.cap_mw)}</span>
        <span className={proposal.feasible ? '' : 'capacity-failed'}>{proposal.feasible ? 'AC pass' : 'AC fail'}</span>
      </div>)}
    </div>}
    <b>Best AC-verified cap</b>
    <strong>{study.best ? `${mw(study.best.cap_mw)} at bus ${study.best.bus}` : 'No feasible proposal found'}</strong>
    <span>{mw(study.requested_mw)} search ceiling · {study.attempts.length} proposals</span>
    {rejected && failed && <p className="capacity-failed">{mw(rejected.cap_mw)} at bus {rejected.bus}{context}. {capacityConstraint(failed)}.</p>}
    <p>Base case and {study.tested_outages.length} selected single-line outages{study.tested_outages.length ? ` (${study.tested_outages.join(', ')})` : ''}.</p>
    <small>{stale ? 'Earlier grid state. Rerun to score the current frame.' : study.scope}</small>
  </div>
}

export default function App() {
  const [state, setState] = useState<GridState | null>(null)
  const [initialTick, setInitialTick] = useState(0)
  const latest = useRef<GridState | null>(null)
  const [frames, setFrames] = useState<GridState[]>([])
  const [scrub, setScrub] = useState<number | null>(null)
  const [selection, setSelection] = useState<Pick>()
  const [prediction, setPrediction] = useState<Forecast | null>(null)
  const [reference, setReference] = useState<{forecast: Forecast; tick: number} | null>(null)
  const [compare, setCompare] = useState<{frames: GridState[]; start: number} | null>(null)
  const [candidateIndex, setCandidateIndex] = useState(0)
  const [metric, setMetric] = useState<Metric>('loading')
  const [error, setError] = useState('')
  const [working, setWorking] = useState(false)
  const busy = useRef(false)
  const [mode, setMode] = useState<Mode>('manual')
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [scenario, setScenario] = useState<Scenario>('baseline')
  const [hidden, setHidden] = useState(false)
  const forecastRequest = useRef<{abort: AbortController; revision: number} | null>(null)
  const previewTimer = useRef<number | undefined>(undefined)
  const previewRequest = useRef<AbortController | null>(null)
  const [previewForecast, setPreviewForecast] = useState<Forecast | null>(null)
  const [study, setStudy] = useState<{id: string; result: CapacityStudy} | null>(null)
  const [studyError, setStudyError] = useState('')
  const [studying, setStudying] = useState(false)
  const [studyTarget, setStudyTarget] = useState<{id: string; revision: number} | null>(null)

  const refresh = useCallback(async (grid: GridState) => {
    forecastRequest.current?.abort.abort()
    const abort = new AbortController()
    forecastRequest.current = {abort, revision: grid.revision}
    setPrediction(null)
    setCandidateIndex(0)
    try {
      const next = await api<Forecast>(`/sessions/${grid.id}/forecast`, {revision:grid.revision}, abort.signal)
      if (!abort.signal.aborted && latest.current?.id === grid.id && latest.current?.revision === next.revision) setPrediction(next)
    } catch (problem) {
      if (!abort.signal.aborted) setError(String(problem instanceof Error ? problem.message : problem))
    }
  }, [])

  const install = useCallback((grid: GridState, reset = false) => {
    latest.current = grid
    setState(grid)
    setScrub(null)
    setFrames(previous => reset ? [grid] : [...previous, grid])
    setPreviewForecast(null)
    void refresh(grid)
  }, [refresh])

  const transact = useCallback(async (work: () => Promise<void>) => {
    if (busy.current) return
    busy.current = true
    setWorking(true)
    setError('')
    window.clearTimeout(previewTimer.current)
    previewRequest.current?.abort()
    setPreviewForecast(null)
    try { await work() }
    catch (problem) {
      setPlaying(false)
      setError(problem instanceof Error ? problem.message : String(problem))
    } finally { busy.current = false; setWorking(false) }
  }, [])

  const reset = useCallback((nextScenario: Scenario) => transact(async () => {
    setPlaying(false)
    const grid = await api<GridState>('/sessions', {scenario: nextScenario, seed:42})
    setScenario(nextScenario); setSelection(undefined); setReference(null); setCompare(null); setHidden(false); setInitialTick(grid.tick)
    setStudy(null); setStudyError(''); setStudyTarget(null)
    install(grid, true)
  }), [install, transact])

  useEffect(() => { void reset('baseline') }, [reset])

  function edit(change: Edit) {
    setPlaying(false)
    const current = latest.current
    if (!current || scrub !== null) return
    void transact(async () => {
      const next = await api<GridState>(`/sessions/${current.id}/edits`, {revision:current.revision,...change})
      setReference(null)
      setCompare(null)
      install(next)
      if (studyTarget) setStudyTarget({id:next.id,revision:next.revision})
    })
  }

  function preview(change: Edit) {
    setPlaying(false)
    window.clearTimeout(previewTimer.current)
    previewRequest.current?.abort()
    const current = latest.current
    if (!current || scrub !== null) return
    previewTimer.current = window.setTimeout(() => {
      const request = new AbortController()
      previewRequest.current = request
      void api<Forecast & {preview?: boolean}>(`/sessions/${current.id}/forecast`, {revision:current.revision,edit:change}, request.signal)
        .then(next => {
          if (!request.signal.aborted && next.preview && latest.current?.id === current.id && latest.current?.revision === next.revision && !busy.current) setPreviewForecast(next)
        }).catch(() => {})
    }, 220)
  }

  function cancelPreview() {
    window.clearTimeout(previewTimer.current)
    previewRequest.current?.abort()
    setPreviewForecast(null)
  }

  const step = useCallback((candidate?: Candidate) => {
    const current = latest.current
    if (!current || scrub !== null) return Promise.resolve()
    return transact(async () => {
      const next = await api<GridState>(`/sessions/${current.id}/step`, {revision:current.revision, mode:candidate ? 'manual' : mode, action:candidate?.action})
      if (prediction?.revision === current.revision && (reference === null || current.tick >= reference.tick + 6)) {
        const action = next.action ?? candidate?.action ?? {kind:'hold'}
        const selected = candidate ?? prediction.candidates.find(({action: proposed}) =>
          proposed.kind === action.kind && proposed.a === action.a && proposed.b === action.b && proposed.value === action.value
        )
        if (selected) setReference({forecast:{...prediction, candidates:[selected]},tick:current.tick})
      }
      install(next)
      if (next.tick === current.tick) setPlaying(false)
    })
  }, [install, mode, prediction, reference, scrub, transact])

  useEffect(() => {
    if (!playing || working || scrub !== null) return
    const timer = window.setTimeout(() => void step(), 1200 / speed)
    return () => window.clearTimeout(timer)
  }, [playing, working, scrub, step, speed])

  function fork() {
    const current = latest.current
    if (!current) return
    setPlaying(false)
    void transact(async () => {
      const next = await api<GridState>(`/sessions/${current.id}/fork`, {revision:current.revision, frame:scrub ?? undefined})
      setReference(null); setCompare(null)
      setStudy(null); setStudyError(''); setStudyTarget(null)
      install(next, true)
    })
  }

  function compareHold() {
    const current = latest.current
    if (!current || scrub !== null) return
    setPlaying(false)
    void transact(async () => {
      let branch = await api<GridState>(`/sessions/${current.id}/fork`, {revision:current.revision})
      const results = []
      for (let i = 0; i < 6; i++) {
        branch = await api<GridState>(`/sessions/${branch.id}/step`, {revision:branch.revision,mode:'manual',action:{kind:'hold'}})
        results.push(branch)
      }
      setReference(null)
      setCompare({frames:results,start:current.tick})
    })
  }

  function runStudy() {
    const current = latest.current
    if (!current || studying || scrub !== null) return
    setPlaying(false)
    setStudyTarget({id:current.id,revision:current.revision})
  }

  useEffect(() => {
    if (!studyTarget) { setStudying(false); return }
    const request = new AbortController()
    setStudyError('')
    setStudying(true)
    const timer = window.setTimeout(async () => {
      try {
        const result = await api<CapacityStudy>(`/sessions/${studyTarget.id}/capacity`, {revision:studyTarget.revision}, request.signal)
        if (!request.signal.aborted) setStudy({id:studyTarget.id,result})
      } catch (problem) {
        if (!request.signal.aborted) setStudyError(problem instanceof Error ? problem.message : String(problem))
      } finally {
        if (!request.signal.aborted) setStudying(false)
      }
    }, 350)
    return () => {window.clearTimeout(timer); request.abort()}
  }, [studyTarget])

  const shown = scrub === null ? state : frames[scrub] ?? state
  const selectedLine = selection?.type === 'line' ? shown?.lines.find(l => l.id === selection.id) : undefined
  const selectedBus = selection?.type === 'bus' ? shown?.buses.find(b => b.id === selection.id) : undefined
  const displayedForecast = scrub === null ? previewForecast ?? prediction : null
  const candidate = displayedForecast?.candidates[candidateIndex]
  const forecastOrigin = previewForecast ? null : reference?.tick
  const plotForecast = ((!previewForecast && reference?.forecast.candidates[0]) || candidate)?.trajectory.map(frame =>
    metric === 'served' ? frame.served : metric === 'voltage' ? low(frame.voltage) : peak(frame.loading)
  ) ?? []
  const observedValue = (frame: GridState) =>
    metric === 'served' ? frame.compute.served : metric === 'voltage' ? low(frame.buses.map(b => b.voltage)) : peak(frame.lines.map(l => l.loading))
  const plotActual = forecastOrigin != null && shown ? Array.from({length: 6}, (_,i) => {
    const frame = frames.find(item => item.tick === forecastOrigin+i+1 && item.revision <= shown.revision)
    return frame ? observedValue(frame) : null
  }) : []
  const plotHold = compare?.start === (forecastOrigin ?? shown?.tick) ? compare?.frames.map(observedValue) ?? [] : []
  const over = shown?.lines.filter(l => l.active && (l.loading ?? 0) > 1).length ?? 0
  const open = shown?.lines.filter(l => !l.active).length ?? 0
  const locked = working || scrub !== null || !shown
  const requestedMax = shown ? Math.ceil(Math.max(shown.compute.requested, 10, shown.buses.reduce((total,b) => total + Math.max(0,b.demand),0) * .1) / 5) * 5 : 50
  const hiddenTarget = selectedBus?.id ?? shown?.compute.bus
  const demandScale = hidden ? (shown?.settings?.hidden_bus === hiddenTarget ? shown?.settings?.hidden_scale : 1) : shown?.settings?.demand_scale
  const demandEdit = (value: number): Edit => ({kind:'demand', value, hidden, target:hidden ? hiddenTarget : undefined})
  const clockText = shown ? clock(shown.tick - initialTick) : 'T+—:—'
  const scoredFrames = compare ? frames.filter((frame,index) => frame.tick > compare.start && frame.tick <= compare.start + 6 &&
    frames.findIndex(other => other.tick === frame.tick) === index) : []
  return <main>
    <section className="workspace">
      <div className="world">
        {shown ? <SceneBoundary><Suspense fallback={<div className="empty">Loading the grid…</div>}><GridScene state={shown} selected={selection} onPick={setSelection} canMove={!locked} onMove={bus => edit({kind:'bus',value:bus})}/></Suspense></SceneBoundary>
        : <div className="empty"><div className="empty-icon">✳</div><h2>{error ? 'Waiting for the simulator' : 'Booting the grid'}</h2><p>{error ? 'Start the Grid Shift API on port 8000, then reset the world.' : 'Loading topology and operating conditions.'}</p>{error && <button onClick={() => void reset(scenario)}>Connect again</button>}</div>}
        <div className="world-overlay">
          <span>TOP VIEW / {shown?.buses.length ?? '—'} BUSES</span><strong>{clockText}</strong>
        </div>
        {shown && <div className="world-stats">
          <div><small>Compute served</small><strong>{mw(shown.compute.served)}</strong></div>
          <div><small>Peak line loading</small><strong className={over ? 'hot' : ''}>{percent(peak(shown.lines.map(l => l.loading)))}</strong></div>
          <div><small>Open branches</small><strong>{open}</strong></div>
        </div>}
        <div className="legend"><span><i className="normal"/>In service</span><span><i className="near"/>Near limit</span><span><i className="critical"/>Over limit</span><span><i className="offline"/>Open</span></div>
        <p className="scene-hint">Drag the coral campus to move it · Drag the map to pan · Scroll to zoom</p>
      </div>
      <aside className="inspector">
        <div className="right-brand">
          <h1 className="wordmark"><span className="brand-symbol" aria-hidden="true">✳</span>Grid Shift<small>TRACK 02</small></h1>
          <span className={`live ${state ? 'online' : ''}`}>{state ? 'Simulator online' : 'Connecting'}</span>
        </div>
        <section className="controls" aria-label="Grid controls">
          {shown && <Mission key={shown.id} grid={shown} disabled={locked} onPause={() => setPlaying(false)} onLine={id => setSelection({type:'line',id})} onBranch={grid => {setPlaying(false);setReference(null);setCompare(null);setInitialTick(grid.tick);install(grid,true)}}/>}
          <div className="panel-head"><span className="eyebrow">01 / SIMULATION</span><h2>Controls</h2></div>
          <div className="scenario-row">
            <select aria-label="Scenario" value={scenario} onChange={event => void reset(event.target.value as Scenario)} disabled={working}>
              <optgroup label="Grid scenarios">
                <option value="baseline">Baseline</option><option value="outage">Line outage</option><option value="hidden">Hidden demand</option>
              </optgroup>
              <optgroup label="Power plant presets">
                <option value="senoko">Singapore · Senoko</option><option value="surry">Virginia · Surry</option><option value="drax">England · Drax</option>
              </optgroup>
            </select>
            <button onClick={() => void reset(scenario)} disabled={working}>Reset</button>
          </div>
          {shown?.site && <div className="site-label">
            <strong>{shown.site.name}</strong>
            <span>{shown.site.region} · {shown.site.technology}</span>
            <small>Site-inspired · Synthetic grid</small>
          </div>}
          <div className="transport">
            <div className="scenario-row">
              <select aria-label="Operator mode" value={mode} onChange={e => {setPlaying(false); setMode(e.target.value as Mode)}}>
                <option value="manual">Manual</option><option value="suggest">WAM suggestions</option><option value="auto">WAM autopilot</option><option value="assisted">Solver assisted</option>
              </select>
              <select aria-label="Playback speed" value={speed} onChange={e => setSpeed(Number(e.target.value))}>
                <option value={.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option><option value={4}>4×</option>
              </select>
            </div>
            <div className="scenario-row transport-buttons">
              <button className="play" disabled={!shown || scrub !== null} onClick={() => setPlaying(!playing)} aria-label={playing ? 'Pause' : 'Play'}>{playing ? 'Ⅱ Pause' : '▶ Play'}</button>
              <button disabled={locked} onClick={() => {setPlaying(false);void step()}}>Step +5m</button>
              <button disabled={!state || working} onClick={fork}>Fork ↗</button>
            </div>
            <label className="scrubber">
              <span>{scrub === null ? 'LIVE' : 'REPLAY'} · {clockText}</span>
              <input aria-label="Timeline" type="range" min={0} max={Math.max(0,frames.length-1)} step={1} value={scrub ?? Math.max(0,frames.length-1)}
                onChange={e => {
                  setPlaying(false); cancelPreview()
                  const next = Number(e.target.value)
                  setScrub(next === frames.length-1 ? null : next)
                }}/>
            </label>
          </div>
          <div className="sliders">
            <Range label="Requested compute" min={0} max={requestedMax} step={.5} suffix=" MW" disabled={locked} value={shown?.compute.requested ?? 0}
              onPreview={value => preview({kind:'compute',value})} onCommit={value => edit({kind:'compute',value})} onCancel={cancelPreview}/>
            <div>
              <Range key={hidden ? `hidden:${hiddenTarget}` : 'global'} label={hidden ? `Bus ${hiddenTarget ?? '—'} hidden demand` : 'Background demand'} min={.8} max={1.4} step={.01} suffix="×" disabled={locked} value={demandScale ?? 1}
                onPreview={value => {if (!hidden) preview(demandEdit(value))}} onCommit={value => edit(demandEdit(value))} onCancel={cancelPreview}/>
              <label className="check"><input type="checkbox" checked={hidden} disabled={locked} onChange={e => {cancelPreview();setHidden(e.target.checked)}}/> Hide demand edit from model</label>
            </div>
          </div>
          {working && <p className="plan" role="status">Updating grid…</p>}
          {error && <p className="error" role="alert">{error}</p>}
        </section>
        <details className="capacity-study"><summary>Legacy selected-outage study</summary>
          <div className="panel-head forecast-head"><span className="eyebrow">02 / TRACK 2</span><h2>Capacity study</h2></div>
          <button disabled={locked || studying} onClick={runStudy}>{studying ? 'Scoring AC power flows…' : 'Run capacity study'}</button>
          {study?.id === shown?.id && study && <CapacityResult study={study.result} stale={study.result.revision !== shown?.revision}/>}
          {studyError && <p className="error" role="alert">{studyError}</p>}
        </details>
        <div className="panel-head forecast-head"><span className="eyebrow">03 / NETWORK</span><h2>Asset inspector</h2></div>
        {selectedLine ? <div className="asset">
          <strong>Line {selectedLine.id} <span className="badge" style={{color:lineStatus(selectedLine).color}}>{lineStatus(selectedLine).name}</span></strong>
          <div><span>Bus {selectedLine.from} → {selectedLine.to}</span><span>{percent(selectedLine.loading)} loaded</span></div>
          <div><span>Flow {mw(selectedLine.flow)}</span><span>Rating {Number(selectedLine.rating).toFixed(0)} A</span></div>
        </div> : selectedBus ? <div className="asset">
          <strong>Bus {selectedBus.id}</strong>
          <div><span>{selectedBus.voltage == null ? '—' : selectedBus.voltage.toFixed(3)} p.u.</span><span>{mw(selectedBus.demand)} demand</span></div>
          <div><span>{mw(selectedBus.generation)} generation</span><button disabled={locked || selectedBus.id === shown?.compute.bus || selectedBus.eligible === false} onClick={() => edit({kind:'bus',value:selectedBus.id})}>Place campus</button></div>
        </div> : <p className="muted">Select a line or substation in the scene to inspect it.</p>}
        {shown && <label className="asset-select"><span>Inspect</span><select aria-label="Select grid asset" value={selection ? `${selection.type}:${selection.id}` : ''} onChange={event => {
          const [type,id] = event.target.value.split(':')
          setSelection(type ? {type:type as Pick['type'], id:Number(id)} : undefined)
        }}>
          <option value="">Choose an asset</option>
          <optgroup label="Substations">{shown.buses.map(b => <option key={b.id} value={`bus:${b.id}`}>Bus {b.id}</option>)}</optgroup>
          <optgroup label="Lines">{shown.lines.map(l => <option key={l.id} value={`line:${l.id}`}>Line {l.id} · {l.from} to {l.to}</option>)}</optgroup>
        </select></label>}
        {selectedLine && <div className="line-control">
          <Range key={selectedLine.id} label={`Line ${selectedLine.id} rating`} min={.7} max={1} step={.01} suffix="×" disabled={locked} value={shown?.settings?.rating_scale[selectedLine.id] ?? 1}
            onPreview={value => preview({kind:'rating',target:selectedLine.id,value})}
            onCommit={value => edit({kind:'rating',target:selectedLine.id,value})} onCancel={cancelPreview}/>
          <button disabled={locked} onClick={() => edit({kind:'line',target:selectedLine.id,value:!selectedLine.active})}>{selectedLine.active ? 'Trip selected line' : 'Restore line'}</button>
        </div>}
        <div className="panel-head forecast-head"><span className="eyebrow">04 / NEXT 30 MIN</span><h2>{previewForecast ? 'Edit preview' : 'Model forecast'}</h2><small>{modelLabel(displayedForecast?.model ?? shown?.model)}</small></div>
        <select className="metric" aria-label="Forecast metric" value={metric} onChange={event => setMetric(event.target.value as Metric)}>
          {Object.entries(metrics).map(([key, item]) => <option key={key} value={key}>{item.label}</option>)}
        </select>
        {scrub !== null && !plotForecast.length
          ? <div className="plot-placeholder">Fork this frame to forecast from it.</div>
          : <Plot forecast={plotForecast} actual={plotActual} hold={plotHold} metric={metric}/>}
        <div className="candidates">
          {displayedForecast?.candidates.slice(0,4).map((item,i) => <button key={i} aria-pressed={i === candidateIndex} className={`candidate ${i === candidateIndex ? 'chosen' : ''}`} onClick={() => {setCandidateIndex(i);setReference(null)}}>
            <span>{String(i+1).padStart(2,'0')}</span><strong>{actionName(item.action)}</strong><i>{percent(peak(item.trajectory.flatMap(frame => frame.loading)))}</i>
          </button>)}
        </div>
        <div className="forecast-buttons">
          <button disabled={locked || !candidate || Boolean(previewForecast)} onClick={() => {setPlaying(false); if (candidate) void step(candidate)}}>Apply action</button>
          <button disabled={locked} onClick={compareHold}>Run hold branch</button>
        </div>
        {Boolean(candidate?.plan?.length) && <p className="plan">Apply runs the first move in a forecast plan.</p>}
        {compare && <div className="comparison"><strong>Same disturbance · {scoredFrames.length}/6 steps observed</strong><span>Hold peak <b>{percent(peak(compare.frames.flatMap(f => f.lines.map(l => l.loading))))}</b></span><span>Live branch peak <b>{percent(peak(scoredFrames.flatMap(f => f.lines.map(l => l.loading))))}</b></span></div>}
        {(shown?.action || shown?.hidden) && <p className="plan">{[shown?.action && actionName(shown.action), shown?.hidden && 'Demand masked from model', shown?.solver_intervened && 'Solver intervened'].filter(Boolean).join(' · ')}</p>}
        {Boolean(shown?.events?.length) && <p className="event">{describeEvent(shown?.events)}</p>}
        <footer><span>SF × MIT HACK DAY</span><span>{shown ? `REV ${shown.revision}` : 'AC POWER FLOW'}</span></footer>
      </aside>
    </section>
  </main>
}
