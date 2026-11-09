import { Component, Suspense, useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react'
import { api, type Line } from './api'
import { GridScene, layout, lineStatus } from './scene'
import type { Dispatch, Frame, MissionEdit, MissionGrid, Pick, Proposal, Region, RunStatus, Study, Violation } from './missionApi'
import { TerrainDiagram } from './RegionalTerrain'
import './mission.css'

const fmt = (value?: number | null, digits = 1) => value == null || !Number.isFinite(value) ? '—' : value.toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits})
const mw = (value?: number | null) => value == null ? '—' : `${fmt(value)} MW`
const processing = (campuses: MissionGrid['campuses']) => campuses.reduce((sum, campus) => sum + campus.processing_mw, 0)
const pct = (value?: number | null) => value == null ? '—' : `${fmt(value * 100, 0)}%`
const clock = (delta: number) => `${delta < 0 ? '−' : '+'}${String(Math.floor(Math.abs(delta) * 5 / 60)).padStart(2, '0')}:${String(Math.abs(delta) * 5 % 60).padStart(2, '0')}`
const lineKinds = /line|branch|thermal|loading|overload/i
const worstLine = (lines: Line[]) => lines.reduce<Line | undefined>((worst, line) => line.active && line.loading != null && (worst?.loading == null || line.loading > worst.loading) ? line : worst, undefined)
const online = (status?: string) => status === 'queued' || status === 'running'
const regionNames: Record<Region, string> = {singapore:'Singapore', virginia:'Northern Virginia', benchmark:'Training benchmark'}
const defaultRegion: Region = 'benchmark'
const regionApi = (_region?: Region) => '/api'
const modeNames: Record<Dispatch,string> = {hold:'Hold', heuristic:'Heuristic', learned:'Learned', assisted:'Assisted', flex:'Flexible workload'}

class MapBoundary extends Component<{children: ReactNode; fallback: ReactNode}, {failed: boolean}> {
  state = {failed:false}
  static getDerivedStateFromError() { return {failed:true} }
  render() { return this.state.failed ? this.props.fallback : this.props.children }
}

function Diagram({grid, selection, limiting, ghost, onPick}: {
  grid: MissionGrid; selection?: Pick; limiting?: number
  ghost?: {allocation: Proposal['allocation']; loading: Frame['loading']}; onPick: (pick: Pick) => void
}) {
  const points = useMemo(() => layout(grid), [grid.buses.length, grid.lines.length, grid.network])
  return <div className="v2-diagram">
    <svg role="group" aria-label="Top-down grid map" viewBox="-13 -9.5 26 19">
      <defs><pattern id="grid-square" width="1" height="1" patternUnits="userSpaceOnUse"><path d="M1 0H0V1" fill="none" stroke="#c8cec2" strokeWidth=".025"/></pattern></defs>
      <rect x="-12.5" y="-9" width="25" height="18" rx=".2" fill={grid.network === 'singapore' ? '#a9ced2' : '#e0e5d9'} stroke="#c1c9bc" strokeWidth=".05"/>
      <rect x="-12.5" y="-9" width="25" height="18" rx=".2" fill="url(#grid-square)"/>
      <TerrainDiagram network={grid.network}/>
      {grid.lines.map((line, index) => {
        const a = points.get(line.from), b = points.get(line.to)
        if (!a || !b) return null
        const hot = limiting === line.id
        const focused = selection?.type === 'line' && selection.id === line.id
        const forecast = ghost?.loading[index]
        const path = `M${a[0]} ${a[2]}L${b[0]} ${b[2]}`
        return <g key={line.id}>
          {hot && <path d={path} stroke="#db7663" strokeWidth=".24" opacity=".5"/>}
          <path d={path} stroke={hot ? '#c4452b' : lineStatus(line).color} strokeWidth={hot || focused ? '.09' : '.042'} strokeDasharray={line.active ? undefined : '.09 .13'}/>
          {forecast != null && forecast >= .8 && <path d={path} stroke="#7554b4" strokeWidth=".065" strokeDasharray=".08 .12"/>}
          <path className="v2-hit" d={path} stroke="transparent" strokeWidth=".2" onClick={() => onPick({type:'line',id:line.id})}><title>Line {line.id}</title></path>
        </g>
      })}
      {grid.buses.map(bus => {
        const point = points.get(bus.id)
        if (!point) return null
        const campus = grid.campuses.find(item => item.bus === bus.id && item.capacity_mw > 0)
        const planned = ghost?.allocation.some(item => item.bus === bus.id && item.mw > 0)
        return <g key={bus.id} className="v2-hit" transform={`translate(${point[0]} ${point[2]})`} onClick={() => onPick({type:'bus',id:bus.id})}>
          <title>Bus {bus.id}{campus ? ', compute campus' : ''}</title>
          {planned && <rect x="-.65" y="-.47" width="1.3" height=".94" rx=".1" fill="#b59be1" fillOpacity=".33" stroke="#7554b4" strokeWidth=".06" strokeDasharray=".12 .09"/>}
          <rect x={campus ? '-.43' : '-.18'} y={campus ? '-.29' : '-.18'} width={campus ? '.86' : '.36'} height={campus ? '.58' : '.36'} rx=".07" fill={campus ? '#3c7667' : '#f2f2e7'} stroke={selection?.type === 'bus' && selection.id === bus.id ? '#c4452b' : '#62786b'} strokeWidth=".06"/>
          {campus?.battery_enabled && <rect x="-.40" y=".39" width={.8 * Math.max(.03, Math.min(1, campus.soc))} height=".08" rx=".035" fill="#849e49"/>}
          <text x="0" y={campus ? '-.38' : '-.25'} textAnchor="middle" fontSize={bus.name ? ".35" : ".24"} fill="#485b50">{bus.name ?? `B${bus.id}`}</text>
        </g>
      })}
    </svg>
  </div>
}

function NumberEdit({label, value, min, max, step, unit, disabled, onSave}: {
  label: string; value: number; min: number; max: number; step: number | "any"; unit: string; disabled: boolean; onSave: (value: number) => void
}) {
  const [draft, setDraft] = useState(String(value))
  const changed = Number(draft) !== value
  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!disabled && changed && draft.trim() && Number.isFinite(Number(draft))) onSave(Number(draft))
  }
  return <form className="v2-field" onSubmit={submit}>
    <label><span>{label}</span><span className="v2-number"><input type="number" min={min} max={max} step={step} value={draft} onChange={event => setDraft(event.currentTarget.value)} disabled={disabled} required/><span>{unit}</span></span></label>
    <button type="submit" disabled={disabled || !changed}>Apply</button>
  </form>
}

function EligibleBuses({grid, eligible, disabled, onSave}: {grid: MissionGrid; eligible: number[]; disabled: boolean; onSave: (buses: number[]) => void}) {
  const [draft,setDraft] = useState(eligible)
  const changed = draft.length !== eligible.length || draft.some(bus => !eligible.includes(bus))
  return <div className="v2-eligible">
    <label htmlFor="v2-eligible-buses">Eligible buses</label>
    <select multiple size={4} id="v2-eligible-buses" value={draft.map(String)} onChange={event => setDraft(Array.from(event.currentTarget.selectedOptions, option => Number(option.value)))}>
      {grid.buses.map(bus => <option key={bus.id} value={bus.id}>{bus.name ?? `Bus ${bus.id}`}</option>)}
    </select>
    <small>Hold ⌘ or Ctrl to select multiple buses.</small>
    <button disabled={disabled || !changed || !draft.length} onClick={() => onSave(draft)}>Apply eligible buses</button>
  </div>
}

function violationText(witness: Violation) {
  const name = witness.kind.replaceAll('_', ' ')
  const loading = lineKinds.test(name)
  const value = loading ? pct(witness.value) : fmt(witness.value, 3)
  const limit = loading ? pct(witness.limit) : fmt(witness.limit, 3)
  return `${loading ? 'Line' : 'Equipment'} ${witness.equipment_id ?? '—'} · ${name} · ${value} / ${limit}`
}

function statusText(proposal: Proposal) {
  if (!proposal.verified) return 'Pending AC'
  if (!proposal.passed) return 'AC fail'
  return proposal.conditional ? 'Conditional' : 'AC pass'
}

function download(study: Study) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(study, null, 2)], {type:'application/json'}))
  const link = document.createElement('a')
  link.href = url
  link.download = `gridshift-study-${study.id}-r${study.revision}.json`
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function frameGrid(grid: MissionGrid, frame: Frame | MissionGrid): MissionGrid {
  if ('lines' in frame) return frame
  return {...grid, tick:frame.tick, violations:frame.violations ?? [], campuses:frame.campuses,
    buses:grid.buses.map((bus, index) => ({...bus,voltage:frame.voltage[index]})),
    lines:grid.lines.map((line, index) => ({...line,loading:frame.loading[index]})),
    compute:{...grid.compute,served:frame.campuses.reduce((sum,campus) => sum + campus.processing_mw, 0)}}
}

export default function MissionV2() {
  const [grid, setGrid] = useState<MissionGrid>()
  const latest = useRef<MissionGrid>(undefined)
  const transaction = useRef<AbortController | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const [study, setStudy] = useState<Study>()
  const [studyPending, setStudyPending] = useState(false)
  const studyRequest = useRef<AbortController>(null)
  const [run, setRun] = useState<RunStatus>()
  const [timeline, setTimeline] = useState<Array<Frame | MissionGrid>>([])
  const [scrub, setScrub] = useState<number>()
  const [selectedId, setSelectedId] = useState<Proposal['id']>()
  const [forecastStep, setForecastStep] = useState(0)
  const [layer, setLayer] = useState<'observed' | 'imagined' | 'verified'>('observed')
  const [selection, setSelection] = useState<Pick>()
  const [stage, setStage] = useState<'setup' | 'study' | 'replay'>('setup')
  const [requestDraft, setRequestDraft] = useState('')
  const [showConstraints, setShowConstraints] = useState(false)
  const evidenceRef = useRef<HTMLDetailsElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const [mode, setMode] = useState<Dispatch>('hold')
  const [playing, setPlaying] = useState(false)
  const [hiddenBus, setHiddenBus] = useState<number>()
  const [batteryBus, setBatteryBus] = useState<number>()
  const [origin, setOrigin] = useState<{proposal: Proposal; study: Study; changed: boolean}>()

  const install = (next: MissionGrid) => {
    const frame = {...next, region:next.region ?? latest.current?.region ?? defaultRegion}
    latest.current = frame
    setGrid(frame)
    setScrub(undefined)
    setLayer('observed')
    setForecastStep(0)
  }

  async function transact(work: (signal: AbortSignal) => Promise<void>, replace = false) {
    if (transaction.current && !replace) return
    transaction.current?.abort()
    const request = new AbortController()
    transaction.current = request
    const timer = window.setTimeout(() => request.abort(), 45_000)
    setPending(true)
    setError('')
    try { await work(request.signal) }
    catch (problem) {
      if (transaction.current !== request) return
      setPlaying(false)
      setError(request.signal.aborted ? 'The request timed out. Reset to start a fresh simulation.' : problem instanceof Error ? problem.message : String(problem))
    } finally {
      window.clearTimeout(timer)
      if (transaction.current === request) {
        transaction.current = null
        setPending(false)
      }
    }
  }

  function stopStudy() {
    studyRequest.current?.abort()
    studyRequest.current = null
    setStudyPending(false)
  }

  function reset(region = grid?.region ?? defaultRegion) {
    setPlaying(false)
    stopStudy()
    void transact(async signal => {
      const next = await api<MissionGrid>('/v2/sessions', {seed:42,region}, signal, regionApi(region))
      if (signal.aborted) return
      setStudy(undefined)
      setSelectedId(undefined)
      setSelection(undefined)
      setMode('hold')
      setTimeline([])
      setHiddenBus(undefined)
      setBatteryBus(undefined)
      setOrigin(undefined)
      setStage('setup')
      setRequestDraft(String(next.compute.requested))
      setShowConstraints(false)
      install({...next,region})
    }, true)
  }

  useEffect(() => {
    reset(defaultRegion)
  }, [])

  useEffect(() => {
    setRun(undefined)
    const controller = new AbortController()
    const refreshRun = () => void api<RunStatus>('/v2/runs/current', undefined, controller.signal, regionApi(grid?.region ?? defaultRegion)).then(setRun).catch(() => {})
    refreshRun()
    const timer = window.setInterval(refreshRun, 10_000)
    return () => { window.clearInterval(timer); controller.abort() }
  }, [grid?.region])

  useEffect(() => () => studyRequest.current?.abort(), [])

  useEffect(() => {
    if (!grid) return
    const controller = new AbortController()
    void api<{frames: Array<Frame | MissionGrid>}>(`/v2/sessions/${grid.id}/timeline`, undefined, controller.signal, regionApi(grid.region))
      .then(result => { if (!controller.signal.aborted) setTimeline(result.frames) })
      .catch(() => {})
    return () => controller.abort()
  }, [grid?.id, grid?.revision])

  useEffect(() => {
    panelRef.current?.scrollTo({top:0})
  }, [stage])

  function startStudy(current: MissionGrid) {
    setStage('study')
    setStudy(undefined)
    setSelectedId(undefined)
    setForecastStep(0)
    setLayer('observed')
    setPlaying(false)
    stopStudy()
    setError('')
    setStudyPending(true)
    const request = new AbortController()
    studyRequest.current = request
    const path = `/v2/sessions/${current.id}/studies`
    void (async () => {
      try {
        const {id} = await api<{id: string}>(path, {revision:current.revision}, request.signal, regionApi(current.region))
        while (!request.signal.aborted) {
          const next = await api<Study>(`/v2/studies/${id}`, undefined, request.signal, regionApi(current.region))
          if (request.signal.aborted || latest.current?.id !== current.id || latest.current?.revision !== current.revision) return
          setStudy(next)
          if (!online(next.status)) break
          await new Promise(resolve => window.setTimeout(resolve, 650))
        }
      } catch (problem) {
        if (!request.signal.aborted) setError(problem instanceof Error ? problem.message : String(problem))
      } finally {
        if (!request.signal.aborted) setStudyPending(false)
      }
    })()
  }

  function edit(change: MissionEdit, runAfter = false) {
    const current = latest.current
    if (!current || scrub !== undefined || transaction.current) return
    setPlaying(false)
    void transact(async signal => {
      const next = await api<MissionGrid>(`/v2/sessions/${current.id}/edits`, {revision:current.revision,...change}, signal, regionApi(current.region))
      if (signal.aborted) return
      install(next)
      setOrigin(previous => previous ? {...previous,changed:true} : undefined)
      if (runAfter) startStudy(next)
      else stopStudy()
    })
  }

  function step() {
    const current = latest.current
    if (!current || scrub !== undefined || transaction.current) return
    void transact(async signal => {
      const next = await api<MissionGrid>(`/v2/sessions/${current.id}/step`, {revision:current.revision,mode}, signal, regionApi(current.region))
      if (signal.aborted) return
      stopStudy()
      install(next)
      if (next.tick === current.tick) setPlaying(false)
    })
  }

  useEffect(() => {
    if (!playing || layer !== 'observed' || pending || scrub !== undefined) return
    const timer = window.setTimeout(step, 750)
    return () => window.clearTimeout(timer)
  }, [playing, layer, pending, scrub, mode, grid?.revision])

  function branch(proposal: Proposal) {
    const current = latest.current
    if (!current || !study || stale || !proposal.verified || study.status !== 'complete' || scrub !== undefined) return
    setPlaying(false)
    void transact(async signal => {
      const next = await api<MissionGrid>(`/v2/sessions/${current.id}/studies/${study.id}/branch`, {revision:current.revision,proposal:proposal.id}, signal, regionApi(current.region))
      if (signal.aborted) return
      setOrigin({proposal,study,changed:false})
      setStage('replay')
      setMode(proposal.policy && (proposal.policy !== 'learned' || run?.available) ? proposal.policy : 'hold')
      setStudy(undefined)
      setTimeline([])
      setSelectedId(undefined)
      setSelection(undefined)
      install(next)
    })
  }

  const stale = !!grid && !!study && study.revision !== grid.revision
  const best = study?.best != null ? typeof study.best === 'object' ? study.best : study.proposals.find(proposal => proposal.id === study.best) : undefined
  const proposal = study?.proposals.find(item => item.id === selectedId) ?? best ?? study?.proposals[0]
  const projected = study ? proposal : origin?.changed ? undefined : origin?.proposal
  const forecast = layer === 'imagined' ? projected?.imagined : projected?.trajectory
  const ghostFrame = stage === 'setup' || layer === 'observed' || stale || scrub !== undefined ? undefined : forecast?.[forecastStep]
  const ghost = ghostFrame && projected ? {allocation:projected.allocation,loading:ghostFrame.loading} : undefined
  const historicalFrame = scrub !== undefined ? timeline[scrub] : undefined
  const shown = grid && historicalFrame ? frameGrid(grid, historicalFrame) : grid
  const selectedLine = selection?.type === 'line' ? shown?.lines.find(line => line.id === selection.id) : undefined
  const busLabel = (id: number) => shown?.buses.find(bus => bus.id === id)?.name ?? `B${id}`
  const selectedBus = selection?.type === 'bus' ? shown?.buses.find(bus => bus.id === selection.id) : undefined
  const selectedCampus = ((layer === 'verified' ? ghostFrame?.campuses : undefined) ?? shown?.campuses)?.find(campus => campus.bus === selectedBus?.id)
  const limitWitness = stale || historicalFrame ? undefined : proposal?.witness
  const witnessId = limitWitness?.equipment_id
  const witnessLine = limitWitness && witnessId != null && lineKinds.test(limitWitness.kind) ? shown?.lines.find(line => line.id === Number(witnessId)) : undefined
  const witnessBus = limitWitness && witnessId != null && /voltage|battery/i.test(limitWitness.kind) ? shown?.buses.find(bus => bus.id === Number(witnessId)) : undefined
  const limiting = witnessLine ?? (shown ? worstLine(shown.lines) : undefined)
  const projectedLimit = ghostFrame && shown ? shown.lines.reduce<{line: Line; loading: number} | undefined>((peak, line, index) => {
    const loading = ghostFrame.loading[index]
    return loading != null && (!peak || loading > peak.loading) ? {line,loading} : peak
  }, undefined) : undefined
  const displayedLimit = witnessLine ?? projectedLimit?.line ?? limiting
  const campusRoster = ((layer === 'verified' ? ghostFrame?.campuses : undefined) ?? shown?.campuses ?? []).filter(campus => campus.capacity_mw > 0)
  const campusCount = shown?.campuses.filter(campus => campus.capacity_mw > 0).length ?? 0
  const startTick = timeline[0]?.tick ?? grid?.tick ?? 0
  const existingHiddenBus = grid?.settings?.hidden_bus
  const targetHiddenBus = hiddenBus ?? (existingHiddenBus != null && existingHiddenBus >= 0 ? existingHiddenBus : grid?.buses[0]?.id)
  const selectedEligible = grid?.eligible_buses ?? grid?.buses.filter(bus => bus.eligible).map(bus => bus.id) ?? []
  const batteryOptions = [...new Set([...selectedEligible, ...(grid?.campuses.map(campus => campus.bus) ?? [])])]
  const batteryTarget = batteryBus !== undefined && batteryOptions.includes(batteryBus) ? batteryBus : batteryOptions[0]
  const batteryEnabled = grid?.campuses.find(campus => campus.bus === batteryTarget)?.battery_enabled ?? !grid?.disabled_batteries?.includes(batteryTarget)
  const evidenceStudy = study ?? origin?.study
  const evidenceProposal = study ? proposal : origin?.proposal
  const baselineFailures = Object.entries(evidenceStudy?.baseline ?? {}).filter(([,witness]) => witness != null)
  const disabled = pending || scrub !== undefined || !grid
  const canForecast = stage !== 'setup' && !studyPending && !stale && scrub === undefined && !!projected && (!!projected.imagined?.length || !!projected.trajectory?.length)
  const complete = !studyPending && study?.status === 'complete' && !stale
  const canOpenBranch = complete && !!proposal?.verified
  const installed = !!origin && !!grid?.campuses.length
  const hasTimeline = stage !== 'setup' && (installed || canForecast || timeline.length > 1)
  const showScrubber = layer === 'observed' ? timeline.length > 1 : !!forecast?.length
  const requested = Number(requestDraft)
  const canStudy = !disabled && !studyPending && Number.isFinite(requested) && requested > 0 && requested <= 5000
  const activeStep = stage === 'setup' ? 0 : stage === 'study' ? 1 : 2
  const latestTraining = Array.isArray(run?.training) ? run.training.at(-1) : undefined
  const modelBackend = run?.inference_backend === 'local' ? 'Local' : run?.inference_backend
  const modelLabel = run?.available ? 'WAM ready' : !run ? 'Checking model' : latestTraining && modelBackend !== 'heuristic' ? 'WAM training' : 'Heuristic mode'
  const studyProgress = !stale && study?.progress === 'Waiting for worker' && study.status === 'running'
    ? study.inference_backend === 'Modal' ? 'Running study on Modal' : 'Running the connection study'
    : !stale ? study?.progress : undefined
  function goToSetup(advanced = false) {
    setPlaying(false)
    stopStudy()
    setStage('setup')
    setScrub(undefined)
    setLayer('observed')
    setShowConstraints(advanced)
  }

  function submitRequest(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!grid || !canStudy) return
    if (requested === grid.compute.requested) startStudy(grid)
    else edit({kind:'compute',value:requested}, true)
  }

  function openEvidence() {
    const details = evidenceRef.current
    if (!details) return
    details.open = true
    details.querySelector('summary')?.focus({preventScroll:true})
    details.scrollIntoView({block:'nearest'})
  }

  const recordedFrames = projected?.trajectory?.length ?? 0
  useEffect(() => {
    if (!playing || layer !== 'verified') return
    if (forecastStep + 1 >= recordedFrames) {
      setPlaying(false)
      return
    }
    const timer = window.setTimeout(() => setForecastStep(forecastStep + 1), 650)
    return () => window.clearTimeout(timer)
  }, [playing, layer, forecastStep, recordedFrames])

  function playRecorded() {
    if (playing && layer === 'verified') {
      setPlaying(false)
      return
    }
    changeLayer('verified')
    setPlaying(true)
  }

  function changeLayer(next: typeof layer) {
    setPlaying(false)
    setLayer(next)
    setForecastStep(0)
    setScrub(undefined)
  }

  function focusLimit() {
    if (displayedLimit) setSelection({type:'line',id:displayedLimit.id})
  }

  function focusWitness() {
    if (witnessLine) setSelection({type:'line',id:witnessLine.id})
    if (witnessBus) setSelection({type:'bus',id:witnessBus.id})
  }

  const diagram = shown && <Diagram grid={shown} selection={selection} limiting={displayedLimit?.id} ghost={ghost} onPick={setSelection}/>

  return <main className="gridshift-v2">
    <header className="v2-topbar">
      <div className="v2-brand"><span className="v2-mark" aria-hidden="true"><svg width="19" height="19" viewBox="0 0 20 20" fill="none"><path d="M11.5 1 4 11h5l-1 8 8-11h-5l.5-7Z" fill="currentColor"/></svg></span><h1>Electrify</h1><span className="v2-version">V2</span></div>
      <div className="v2-top-state">
        <button className={`v2-model ${run?.available ? 'ready' : ''}`} onClick={openEvidence} title="View model and study evidence">
          <i aria-hidden="true"/>{modelLabel}{run?.available && modelBackend && <small> · {modelBackend}</small>}
        </button>
        <button className="v2-reset" onClick={() => reset()} title="Start a fresh simulation">Reset</button>
      </div>
    </header>

    <div className="v2-workspace">
      <section className={`v2-world ${hasTimeline ? 'has-timeline' : ''}`} aria-label="Grid viewport">
        <div className="v2-map">
          {shown ? <MapBoundary fallback={diagram}><Suspense fallback={diagram}><GridScene state={shown} selected={selection} campuses={shown.campuses} limiting={displayedLimit?.id} ghost={ghost} onPick={setSelection} fallback={diagram}/></Suspense></MapBoundary>
            : <div className="v2-empty"><strong>{error ? 'Simulator unavailable' : 'Connecting to the grid'}</strong><span>{error || 'Loading the network and operating conditions.'}</span></div>}
        </div>

        <div className="v2-map-top">
          <div className="v2-map-title"><strong>{shown?.region_details?.name ?? 'Grid map'}</strong><small>{shown ? `${shown.buses.length} buses · ${shown.lines.length} lines` : 'Loading topology'}</small></div>
          {displayedLimit && <button className="v2-map-limit" onClick={focusLimit}>
            <span>{witnessLine ? 'Study bottleneck' : ghostFrame ? 'Projected peak' : 'Peak load'}</span>
            <b>L{displayedLimit.id}{witnessLine ? '' : ` · ${pct(projectedLimit?.loading ?? displayedLimit.loading)}`}</b><span aria-hidden="true">↗</span>
          </button>}
        </div>
        {selection && (selectedLine || selectedBus) && <div className="v2-map-inspect">
          <button className="v2-close" aria-label="Close map selection" onClick={() => setSelection(undefined)}>×</button>
          <span className="v2-eyebrow">{selectedLine ? 'TRANSMISSION' : selectedCampus ? 'COMPUTE CAMPUS' : 'GRID BUS'}</span>
          <strong>{selectedLine ? `Line ${selectedLine.id}` : selectedBus?.name ?? `Bus ${selectedBus?.id}`}</strong>
          {selectedLine ? <p>{busLabel(selectedLine.from)} → {busLabel(selectedLine.to)} · {selectedLine.active ? `${pct(selectedLine.loading)} loading` : 'Open'}<br/>{fmt(selectedLine.rating)} A rating</p>
            : <p>{fmt(selectedBus?.voltage, 3)} p.u. · {mw(selectedBus?.demand)} demand{selectedCampus && <><br/>{mw(selectedCampus.processing_mw)} processing · {selectedCampus.battery_enabled ? `${pct(selectedCampus.soc)} battery` : 'Battery disabled'}<br/>{fmt(selectedCampus.backlog_mwh, 2)} MWh backlog</>}</p>}
          <button disabled={pending || studyPending} onClick={() => goToSetup(true)}>Edit constraints ↗</button>
        </div>}
        <div className="v2-legend" aria-label="Map legend">
          <span><i/> In service</span><span><i className="amber"/> Near limit</span><span><i className="hot"/> Bottleneck</span><span><i className="open"/> Open</span>
          {campusCount > 0 && <span><i className="campus"/> Campus</span>}
          {ghost && <span><i className="ghost"/> {layer === 'imagined' ? 'WAM ghost' : 'AC projection'}</span>}
        </div>

        {hasTimeline && <section className="v2-timeline" aria-label="Simulation timeline">
          <div className="v2-timeline-top">
            <div><span className="v2-eyebrow">{historicalFrame ? 'REPLAY' : 'TIMELINE'}</span><strong>{shown ? `T${clock((ghostFrame?.tick ?? shown.tick) - startTick)}` : 'T+—:—'}</strong></div>
            {canForecast && <div className="v2-layers" role="group" aria-label="Timeline layer">
              <button aria-pressed={layer === 'observed'} onClick={() => changeLayer('observed')}>Observed</button>
              {!!projected?.imagined?.length && <button aria-pressed={layer === 'imagined'} onClick={() => changeLayer('imagined')}>WAM ghost</button>}
              {!!projected?.trajectory?.length && <button aria-pressed={layer === 'verified'} onClick={() => changeLayer('verified')}>AC forecast</button>}
            </div>}
          </div>
          {showScrubber && <div className="v2-timeline-controls">
            {showScrubber && <label className="v2-slider"><span className="v2-sr">{layer === 'observed' ? 'Replay frame' : 'Forecast frame'}</span><input type="range" aria-label={layer === 'observed' ? 'Replay frame' : 'Forecast frame'} min={0} max={Math.max(0, (layer === 'observed' ? timeline.length : forecast?.length ?? 0) - 1)} value={layer === 'observed' ? scrub ?? Math.max(0,timeline.length - 1) : forecastStep} disabled={pending} onChange={event => {
              const index = Number(event.currentTarget.value)
              setPlaying(false)
              if (layer === 'observed') setScrub(index === timeline.length - 1 ? undefined : index)
              else setForecastStep(index)
            }}/></label>}
            {scrub !== undefined && <button onClick={() => setScrub(undefined)}>Live</button>}
          </div>}
        </section>}
      </section>

      <aside className="v2-sidebar" aria-label="Mission controls">
        <ol className="v2-steps" aria-label="Study progress">
          {['Setup','Results','Replay'].map((label, index) => <li key={label} aria-current={index === activeStep ? 'step' : undefined} data-complete={index < activeStep}>
            <span aria-hidden="true">{index < activeStep ? '✓' : index + 1}</span>{label}
          </li>)}
        </ol>
        <div className="v2-panel" ref={panelRef}>
          {error && grid && <p className="v2-error" role="alert">{error}</p>}
          {stage === 'setup' && <>
            <h2>Connect compute</h2>
            <p className="v2-copy">Enter a request to see where it can connect and what limits it.</p>
            <label className="v2-region-picker">Region
              <select value={grid?.region ?? defaultRegion} disabled={pending} onChange={event => reset(event.currentTarget.value as Region)}>
                {Object.entries(regionNames).map(([id, name]) => <option key={id} value={id}>{name}</option>)}
              </select>
            </label>
            {grid?.region_details && <p className="v2-copy">Synthetic regional case · WAM transfer</p>}
            <form className="v2-setup-form" onSubmit={submitRequest}>
              <label htmlFor="v2-requested">Requested compute</label>
              <span className="v2-number"><input id="v2-requested" type="number" min={0} max={5000} step="any" required value={requestDraft} onChange={event => setRequestDraft(event.currentTarget.value)} disabled={disabled}/><span>MW</span></span>
              {installed && requested !== grid?.compute.requested && <small className="v2-copy">Changing the request clears the installed campuses from this branch.</small>}
              <button className="v2-primary" type="submit" disabled={!canStudy}>Run connection study <span aria-hidden="true">→</span></button>
            </form>
            <details className="v2-disclosure" open={showConstraints} onToggle={event => setShowConstraints(event.currentTarget.open)}>
              <summary>Advanced constraints</summary>
              {grid && <div className="v2-constraints" key={`${grid.id}:${grid.revision}`}>
                <fieldset disabled={disabled}><legend>Sites and batteries</legend>
                  <EligibleBuses grid={grid} eligible={selectedEligible} disabled={disabled} onSave={value => edit({kind:'eligible',value})}/>
                  {!!batteryOptions.length && <><label className="v2-select">Battery bus <select value={batteryTarget} onChange={event => setBatteryBus(Number(event.currentTarget.value))}>{batteryOptions.map(bus => <option key={bus} value={bus}>B{bus}</option>)}</select></label>
                  <label className="v2-check"><input type="checkbox" checked={batteryEnabled} onChange={event => edit({kind:'battery',value:event.currentTarget.checked,target:batteryTarget})}/> Battery availability</label></>}
                </fieldset>
                <fieldset disabled={disabled}><legend>Demand</legend>
                  <NumberEdit label="Background demand" value={grid.settings?.demand_scale ?? 1} min={.1} max={2} step="any" unit="×" disabled={disabled} onSave={value => edit({kind:'demand',value})}/>
                  <label className="v2-select">Hidden demand at bus <select value={targetHiddenBus ?? ''} onChange={event => setHiddenBus(Number(event.currentTarget.value))}>{grid.buses.map(bus => <option key={bus.id} value={bus.id}>B{bus.id}</option>)}</select></label>
                  <NumberEdit key={targetHiddenBus} label="Hidden demand scale" value={targetHiddenBus === grid.settings?.hidden_bus ? grid.settings?.hidden_scale ?? 1 : 1} min={.1} max={2} step="any" unit="×" disabled={disabled || targetHiddenBus == null} onSave={value => edit({kind:'demand',value,hidden:true,target:targetHiddenBus})}/>
                </fieldset>
                <fieldset disabled={disabled}><legend>Transmission</legend>
                  <label className="v2-select">Branch <select value={selectedLine?.id ?? ''} onChange={event => setSelection(event.currentTarget.value ? {type:'line',id:Number(event.currentTarget.value)} : undefined)}><option value="">Select a line</option>{grid.lines.map(line => <option key={line.id} value={line.id}>L{line.id} · {busLabel(line.from)} → {busLabel(line.to)}</option>)}</select></label>
                  {selectedLine && <>
                    <label className="v2-check"><input type="checkbox" checked={!selectedLine.active} onChange={event => edit({kind:'line',value:!event.currentTarget.checked,target:selectedLine.id})}/> Branch outage</label>
                    <NumberEdit key={selectedLine.id} label="Rating scale" value={grid.settings?.rating_scale[grid.lines.findIndex(line => line.id === selectedLine.id)] ?? 1} min={.1} max={2} step="any" unit="×" disabled={disabled} onSave={value => edit({kind:'rating',value,target:selectedLine.id})}/>
                  </>}
                </fieldset>
              </div>}
            </details>
          </>}

          {stage === 'study' && <>
            <button className="v2-back" disabled={pending || studyPending} onClick={() => goToSetup()}>← Back to setup</button>
            <h2>{studyPending ? 'Checking the grid' : 'Study results'}</h2>
            <div className="v2-request"><span>Requested compute</span><strong>{mw(grid?.compute.requested)}</strong></div>
            {studyPending && <div className="v2-progress" role="status">
              <span>{studyProgress || 'Starting a new study'}</span>
              <small>{!stale && study?.solves ? `${study.solves} AC solves` : 'Verifying allocations with AC power flow'}</small>
            </div>}
            {!studyPending && !complete && !stale && study && <p className="v2-copy v2-baseline" role="status">{study.progress}</p>}
            {complete && !proposal && <p className="v2-copy v2-baseline">{study.progress}</p>}
            {!studyPending && !complete && <button className="v2-primary" disabled={!canStudy} onClick={() => grid && startStudy(grid)}>Retry study →</button>}
            {complete && proposal && <section className="v2-options" aria-label="Connection proposals">
              <h3>Choose a proposal</h3>
              {study?.proposals.map(item => <button className="v2-proposal" key={item.id} aria-pressed={proposal.id === item.id} onClick={() => {setPlaying(false);setSelectedId(item.id);setForecastStep(0);if (layer === 'imagined' && !item.imagined?.length || layer === 'verified' && !item.trajectory?.length) setLayer('observed')}}>
                <span><b>{item.source.replaceAll('_',' ')}</b><small>{mw(item.accepted_mw)} · {item.allocation.map(site => busLabel(site.bus)).join(', ') || 'No sites'}</small></span>
                <span className={`v2-outcome ${item.passed && item.conditional ? 'conditional' : item.verified && !item.passed ? 'failed' : ''}`}>{statusText(item)}</span>
              </button>)}
              {!study?.proposals.some(item => item.passed) && <p className="v2-copy">None of the tested proposals passed. Try a smaller request or watch the recorded AC trace to inspect a failure.</p>}
              {recordedFrames > 0 && <button className="v2-secondary" onClick={playRecorded}>{playing && layer === 'verified' ? 'Pause recorded AC trace' : 'Play recorded AC trace'}</button>}
              {proposal.witness && <button className="v2-witness" disabled={!witnessLine && !witnessBus} onClick={focusWitness}>{violationText(proposal.witness)}{(witnessLine || witnessBus) && ' ↗'}</button>}
              {canOpenBranch && <>
                {proposal.conditional && proposal.passed && <p className="v2-copy v2-condition-note">Passes the AC cases where existing service was within limits. See study details for baseline failures.</p>}
                <button className="v2-primary" disabled={disabled} onClick={() => branch(proposal)}>{proposal.passed ? 'Open branch for replay' : 'Replay failed proposal'} <span aria-hidden="true">→</span></button>
              </>}
            </section>}
          </>}

          {stage === 'replay' && origin && <>
            <button className="v2-back" disabled={pending || studyPending} onClick={() => goToSetup()}>← Back to setup</button>
            <h2>Replay the branch</h2>
            <div className="v2-branch-note"><b>{mw(origin.proposal.accepted_mw)} installed</b><span>{mw(shown ? processing(shown.campuses) : undefined)} processing · {campusCount} campus{campusCount === 1 ? '' : 'es'}</span>{origin.changed && <span>Branch constraints edited. Source projections archived.</span>}</div>
            <label className="v2-select">Dispatch policy <select value={mode} onChange={event => setMode(event.currentTarget.value as Dispatch)}>{Object.entries(modeNames).map(([key,label]) => <option value={key} key={key} disabled={(key === 'learned' || key === 'assisted') && !run?.available}>{label}</option>)}</select></label>
            <div className="v2-replay-actions">
              <button className="v2-primary" disabled={disabled && !playing} onClick={() => {changeLayer('observed');setPlaying(!(playing && layer === 'observed'))}}>{playing && layer === 'observed' ? 'Ⅱ Pause' : '▶ Play simulation'}</button>
              <button className="v2-secondary" disabled={disabled} onClick={() => {changeLayer('observed');step()}}>Step +5m</button>
            </div>
            {!origin?.changed && recordedFrames > 0 && <button className="v2-secondary" onClick={playRecorded}>{playing && layer === 'verified' ? 'Pause recorded AC trace' : 'Play recorded AC trace'}</button>}
            {!!campusRoster.length && <div className="v2-campuses" aria-label="Campus telemetry">{campusRoster.map(campus => <button key={campus.bus} onClick={() => setSelection({type:'bus',id:campus.bus})}>
              <span><b>{busLabel(campus.bus)}</b><strong>{mw(campus.processing_mw)} / {mw(campus.capacity_mw)}</strong></span>
              <small>{campus.battery_enabled ? `${pct(campus.soc)} battery` : 'Battery disabled'} · {fmt(campus.backlog_mwh, 2)} MWh backlog</small>
            </button>)}</div>}
          </>}

          <details className="v2-disclosure" ref={evidenceRef}>
            <summary>{evidenceStudy ? 'Study details' : 'Model details'}</summary>
            <dl className="v2-facts">
              <div><dt>Revision</dt><dd>{shown?.revision ?? '—'}</dd></div>
              <div><dt>Seed</dt><dd>{grid?.seed ?? '—'}</dd></div>
              {grid?.region_details && <div><dt>Demand reference</dt><dd><a href={grid.region_details.source} target="_blank" rel="noreferrer">{mw(grid.region_details.peak_mw)} · {grid.region_details.reference}</a></dd></div>}
              {grid?.network === 'singapore' && <div><dt>Geography</dt><dd><a href="https://www.geoboundaries.org/api/current/gbOpen/SGP/ADM0/" target="_blank" rel="noreferrer">URA via geoBoundaries · ODbL 1.0</a></dd></div>}
              {grid?.network === 'virginia' && <div><dt>Geography</dt><dd><a href="https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.2025.html" target="_blank" rel="noreferrer">U.S. Census · 2025 counties and TIGER water</a></dd></div>}
              <div><dt>Synthetic grid</dt><dd>{grid?.network ?? '—'}</dd></div>
              <div><dt>Checkpoint</dt><dd>{run?.model_version ?? '—'}</dd></div>
              <div><dt>Status</dt><dd>{modelLabel}</dd></div>
              <div><dt>Inference backend</dt><dd>{modelBackend && modelBackend !== 'heuristic' ? run?.available ? modelBackend : `${modelBackend} · unavailable` : '—'}</dd></div>
              <div><dt>Training stage</dt><dd>{latestTraining?.stage ?? '—'}</dd></div>
              <div><dt>Training step</dt><dd>{fmt(latestTraining?.step,0)}</dd></div>
            </dl>
            {evidenceProposal?.metrics && <section className="v2-evidence-block"><h3>{evidenceProposal.source.replaceAll('_',' ')} outcome</h3><p>{statusText(evidenceProposal)}</p><dl className="v2-facts">{[
              ['Completed work','completed_mwh'],['Immediate loss','immediate_lost_mwh'],['Deadline misses','deadline_miss_mwh'],['Existing demand lost','existing_lost_mwh'],['Terminal battery','battery_terminal_mwh'],
            ].map(([label, key]) => <div key={key}><dt>{label}</dt><dd>{typeof evidenceProposal.metrics?.[key] === 'number' ? `${fmt(evidenceProposal.metrics[key] as number, 2)} MWh` : '—'}</dd></div>)}</dl></section>}
            {!!shown?.violations.length && <section className="v2-evidence-block"><h3>Observed violations</h3>{shown.violations.slice(0,8).map((witness,index) => <p key={index}>{violationText(witness)}</p>)}</section>}
            {evidenceStudy && <section className="v2-evidence-block">
              <h3>Study log</h3>
              {(stale || !study) && <p className="v2-hot">{!study ? 'Source branch study' : 'Archived evidence'} · revision {evidenceStudy.revision}.</p>}
              <p>{evidenceStudy.scope}</p>
              {evidenceStudy.model_version && <p>Study model · {evidenceStudy.model_version}{evidenceStudy.inference_backend && ` · ${evidenceStudy.inference_backend} inference`}</p>}
              {!!evidenceStudy.tested_cases?.length && <p>Tested cases · {evidenceStudy.tested_cases.join(', ')}</p>}
              {!!baselineFailures.length && <ul className="v2-activity">{baselineFailures.map(([name,witness]) => <li key={name}>{name} · {witness && violationText(witness)}</li>)}</ul>}
              {evidenceProposal?.grade && <p>{evidenceProposal.grade.failed_cases} failed cases · {evidenceProposal.grade.violations} violations across the evaluated cases{evidenceProposal.parent_id != null && ` · revised proposal ${Number(evidenceProposal.parent_id)+1}`}</p>}
              {evidenceProposal?.cases && <ul className="v2-activity">{evidenceProposal.cases.map(item => <li key={item.id}>{item.id} · {item.passed ? 'AC pass' : item.witness ? violationText(item.witness) : 'AC fail'}</li>)}</ul>}
              {Array.isArray(evidenceStudy.activity) && <ol className="v2-activity">{evidenceStudy.activity.slice(-10).reverse().map((item,index) => <li key={index}>{String(item.message ?? item.kind ?? item)}</li>)}</ol>}
              <button className="v2-secondary" onClick={() => download(evidenceStudy)}>Export study JSON ↓</button>
            </section>}
          </details>
        </div>
        <div className="v2-sidebar-footer"><span>5 MIN / STEP</span><span>{shown ? `REV ${shown.revision}` : 'REV —'}</span></div>
      </aside>
    </div>
  </main>
}
