import { useEffect, useRef, useState } from 'react'
import { api, type CapacityCase, type GridState } from './api'

type Proposal = {id:number; bus:number; cap_mw:number; source:string; new_failures:number; unresolved:number; worsened:number; feasible:boolean; witness:CapacityCase|null; imagined?:{trajectory:{event:boolean}[]}|null}
type MissionData = {id:string; revision:number; status:string; stage:string; progress:string; requested_mw:number; proposals:Proposal[]; preexisting?:Array<number|null>; outages?:number[]; best?:Proposal|null; solves?:number; duration_s?:number; scope?:string; model?:string}
const mw = (v:number) => `${v.toFixed(1)} MW`
function reason(c:CapacityCase) {
  const context = c.outage == null ? 'Intact grid' : `Outage of line ${c.outage}`
  if(c.failure) return `${context}: AC solve failed.`
  if(c.lost > .001) return `${context}: ${mw(c.lost)} of existing demand disconnected.`
  if(c.loading > 1) return `${context}: line ${c.worst_line} reaches ${(100*c.loading).toFixed(1)}% loading.`
  if(c.min_voltage < .9) return `${context}: voltage falls to ${c.min_voltage.toFixed(3)} p.u.`
  return `${context}: electrical or delivery limit failed. See exported evidence.`
}

export function Mission({grid, disabled, onPause, onBranch, onLine}: {grid:GridState; disabled:boolean; onPause:()=>void; onBranch:(g:GridState)=>void; onLine:(id:number)=>void}) {
  const [enabled,setEnabled] = useState(false)
  const [follow,setFollow] = useState(true)
  const [nonce,setNonce] = useState(0)
  const [job,setJob] = useState<MissionData|null>(null)
  const [error,setError] = useState('')
  const [branching,setBranching] = useState(false)
  const [previous,setPrevious] = useState<number|null>(null)
  const lastComplete = useRef<MissionData|null>(null)
  const revision = follow ? grid.revision : 0
  useEffect(() => {
    if(!enabled) return
    let active = true
    const controller = new AbortController()
    let poll:ReturnType<typeof setTimeout>
    const start = setTimeout(async () => {
      setError('')
      try {
        const result = await api<{id:string}>(`/sessions/${grid.id}/missions`,{revision:grid.revision},controller.signal)
        const update = async () => {
          try {
            const data = await api<MissionData>(`/missions/${result.id}`,undefined,controller.signal)
            if(!active) return
            setJob(data)
            if(data.status === 'complete') {
              if(lastComplete.current?.revision !== data.revision) setPrevious(lastComplete.current?.best?.cap_mw ?? null)
              lastComplete.current = data
            } else if(data.status === 'queued' || data.status === 'running') poll = setTimeout(update,800)
          } catch(e) { if(active) setError(String(e)) }
        }
        await update()
      } catch(e) { if(active) setError(String(e)) }
    },600)
    return () => {active=false;controller.abort();clearTimeout(start);clearTimeout(poll)}
  },[enabled,grid.id,revision,nonce])
  const stale = job?.revision !== grid.revision
  const running = enabled && (!job || (stale && follow) || ['queued','running'].includes(job.status)) && !error
  const stages = ['baseline','imagine','verify','complete']
  const stage = stages.indexOf(job?.stage ?? '')
  async function branch(p:Proposal) {
    onPause();setBranching(true);setError('')
    try {onBranch(await api<GridState>(`/sessions/${grid.id}/missions/${job!.id}/branch`,{revision:grid.revision,proposal:p.id}))}
    catch(e) {setError(String(e))} finally {setBranching(false)}
  }
  function download() {
    const url=URL.createObjectURL(new Blob([JSON.stringify(job,null,2)],{type:'application/json'}))
    const a=document.createElement('a');a.href=url;a.download=`gridshift-study-rev-${job?.revision}.json`;a.click();URL.revokeObjectURL(url)
  }
  const best = job?.best
  return <section className="mission" aria-label="Connection mission">
    <span className="eyebrow">CONNECTION MISSION</span>
    <h2>Find room for {mw(grid.compute.requested)}</h2>
    <p>Test the request. Find the bottleneck. Explore a better connection.</p>
    <button className="mission-start" disabled={disabled || branching || (running && !stale)} onClick={()=>{onPause();setEnabled(true);setNonce(n=>n+1)}}>{running ? 'Investigating the grid…' : enabled ? 'Re-run connection mission' : 'Find a connection plan →'}</button>
    <label className="check"><input type="checkbox" checked={follow} onChange={e=>setFollow(e.target.checked)}/> Re-study after constraint edits</label>
    {enabled && <ol className="mission-stages">{['Baseline','Imagine','Verify','Compare'].map((name,i)=><li key={name} className={i===stage?'current':i<stage?'done':''}>{name}</li>)}</ol>}
    {job && <>
      <p role="status" className="mission-status">{stale ? 'Grid changed. Earlier evidence is stale.' : job.progress}</p>
      <small>{job.solves ?? 0} AC solves · {job.outages?.length ?? '…'} branch outages · revision {job.revision}</small>
      {!!job.preexisting?.length && <p className="mission-warning">{job.preexisting.length} baseline cases already fail without new compute. These remain in the report; no unconditional capacity claim.</p>}
      {best && <div className="mission-best"><span>{best.feasible ? 'Verified in all tested cases' : 'Best option without new failures or measured worsening'}</span><strong>{mw(best.cap_mw)} · Bus {best.bus}</strong>{previous !== null && <small>{best.cap_mw-previous >= 0 ? '+' : ''}{(best.cap_mw-previous).toFixed(1)} MW versus the previous study</small>}</div>}
      <div className="mission-options">{job.proposals.map(p=><article key={p.id} className={p.id===best?.id?'best':''}>
        <div><b>{p.source}</b><span>{mw(p.cap_mw)} · B{p.bus}</span></div>
        <small>{p.feasible ? 'All tested cases pass' : `${p.new_failures} new failures · ${p.unresolved} unresolved · ${p.worsened} worsened baseline cases`}</small>
        {p.imagined && <small>WAM: {p.imagined.trajectory.filter(x=>x.event).length}/6 future steps predict violations. Outage checks below are independent.</small>}
        {p.witness && <button className="mission-witness" disabled={p.witness.failure || p.witness.loading<=1} onClick={()=>onLine(p.witness!.worst_line)}>{reason(p.witness)}{!p.witness.failure && p.witness.loading>1 ? ' Locate →' : ''}</button>}
        {job.status==='complete' && <button disabled={disabled||stale||branching} onClick={()=>void branch(p)}>Explore this branch ↗</button>}
      </article>)}</div>
      {job.status==='complete' && <><button onClick={download}>Export study evidence ↓</button><p className="plan">{job.scope}</p><small>WAM: {job.model}. Engineering alternatives are generated by explicit search.</small></>}
    </>}
    {error && <p className="error" role="alert">{error}</p>}
  </section>
}
