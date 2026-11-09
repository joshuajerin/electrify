"""Revision-bound capacity planning API."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path
from threading import Event, RLock
from uuid import uuid4
import time

import numpy as np
from fastapi import APIRouter, HTTPException

from .planner_v2 import candidates, flex_step, ordered, study
from .sim_v2 import ResourceSimulator
from .regional_cases import DEFAULT_REQUEST, LAYOUTS, REGIONS

router = APIRouter(prefix="/v2")
lock = RLock()
worker = ThreadPoolExecutor(max_workers=1)
sessions = {}
jobs = {}
checkpoint = Path(os.getenv("GRIDSHIFT_V2_CHECKPOINT","artifacts/gridshift-v2-wam-001/checkpoint.pt"))
modal_app = os.getenv("GRIDSHIFT_MODAL_APP", "gridshift-inference-v2")
modal_studies_left = int(os.getenv("GRIDSHIFT_MODAL_STUDIES","0"))
modal_steps_left = int(os.getenv("GRIDSHIFT_MODAL_STEPS","0"))
modal_until = float(os.getenv("GRIDSHIFT_MODAL_UNTIL","0"))


def modal_ready():
    return checkpoint.is_file() and time.time()<modal_until


def local_model_ready():
    return os.getenv("GRIDSHIFT_LOCAL_WAM") == "1" and checkpoint.is_file()


@lru_cache(maxsize=1)
def local_model():
    import torch
    from .model_v2 import load_resource_checkpoint
    torch.set_num_threads(2)
    return load_resource_checkpoint(checkpoint, "cpu")


def read_session(id, body=None):
    session = sessions.get(id)
    if session is None:
        raise HTTPException(404,"Unknown session")
    if body is not None and body.get("revision")!=session["revision"]:
        raise HTTPException(409,"Scenario changed. Use the latest revision.")
    return session


def publish_frame(session):
    s = session["sim"]
    o = s.observation
    region = REGIONS.get(s.name)
    state = dict(id=session["id"],revision=session["revision"],tick=int(s.tick),seed=session["seed"],
                 network=s.name,region=s.name if region else "benchmark",
                 region_details=dict(name=region.name,peak_mw=region.peak_mw,reference=region.reference,source=region.source) if region else None,
                 model_version=checkpoint.parent.name if checkpoint.is_file() else "heuristic",
                 compute=dict(bus=s.eligible[0],requested=s.request,cap=sum(c["capacity_mw"] for c in s.campuses.values()),served=o["served"]),
                 campuses=s.campus_view(),violations=o.get("violations",s.violations(o)),
                 disabled_batteries=sorted(s.disabled_batteries),eligible_buses=list(s.eligible),
                 settings=dict(demand_scale=s.demand_scale,rating_scale=s.rating_scale.tolist(),
                               hidden_bus=s.hidden_bus,hidden_scale=s.hidden_scale),
                 buses=[dict(id=i,voltage=float(o["voltage"][i]),demand=float(o["demand"][i]),generation=float(o["generation"][i]),
                             eligible=i in s.eligible, name=region.buses[i] if region else None,
                             position=LAYOUTS[s.name][i] if region else None) for i in range(s.n)],
                 lines=[dict(id=i,**{"from":int(a),"to":int(b)},active=bool(o["active"][i]),
                             loading=float(o["loading"][i]),flow=float(o["flow"][i]),
                             rating=float(s.meta["ratings"][i]*s.rating_scale[i])) for i,(a,b) in enumerate(s.edges.T)],
                 events=[v["kind"] for v in o.get("violations",[])],
                 action=o.get("executed_action"),mode=session.get("mode","heuristic"),
                 rejected=bool(o.get("rejected",False)),clipped=bool(o.get("clipped",False)))
    session["frames"].append(state)
    return state


def new_session(sim,seed,mode="heuristic"):
    id = uuid4().hex
    session = dict(id=id,sim=sim,revision=0,seed=seed,frames=[],mode=mode)
    sessions[id] = session
    return publish_frame(session)


def cancel_jobs(id):
    for job in jobs.values():
        if job["session_id"]==id and job["status"] in ("queued","running"):
            job["cancel"].set()


@router.post("/sessions")
def create(body: dict):
    try:
        region = body.get("region", "benchmark")
        if region != "benchmark" and region not in REGIONS:
            raise ValueError("Unknown region")
        request = float(body.get("requested_mw",DEFAULT_REQUEST.get(region,40)))
        seed = int(body.get("seed",42))
        if not np.isfinite(request) or not 0<=request<=5000 or not 0<=seed<2**32:
            raise ValueError()
        with lock:
            network = {"name":region} if region in REGIONS else {}
            sim = ResourceSimulator(smoke=os.getenv("GRIDSHIFT_SMOKE","1")=="1",seed=seed,request=request,**network)
            for tick in range(1,12):
                sim.tick=tick
                sim.observe()
            if "eligible_buses" in body:
                sim.scenario_edit("eligible",body["eligible_buses"])
            return new_session(sim,seed)
    except (ValueError,TypeError,IndexError) as exc:
        raise HTTPException(422,"Invalid scenario request") from exc


@router.get("/sessions/{id}")
def state(id: str):
    with lock:
        return read_session(id)["frames"][-1]


@router.get("/sessions/{id}/timeline")
def timeline(id: str):
    with lock:
        return dict(frames=list(read_session(id)["frames"]))


@router.post("/sessions/{id}/edits")
def edit(id: str, body: dict):
    with lock:
        session = read_session(id,body)
        sim = session["sim"].fork()
        try:
            kind = body["kind"]
            value = body["value"]
            if kind=="bus":
                kind,value = "eligible",[int(value)]
            if kind=="compute" and sim.campuses:
                sim = fresh_study(sim)
            sim.scenario_edit(kind,value,int(body.get("target",0)),bool(body.get("hidden",False)))
        except (ValueError,TypeError,IndexError,KeyError) as exc:
            raise HTTPException(422,str(exc)) from exc
        cancel_jobs(id)
        session["sim"] = sim
        session["revision"] += 1
        return publish_frame(session)


def fresh_study(sim):
    if not sim.campuses:
        return sim
    sim.net.load.drop(index=[c[k] for c in sim.campuses.values() for k in ("load","storage")],inplace=True)
    sim.campuses.clear()
    sim.duration = 0
    sim.queue[:] = 0
    sim.observe(replace=True)
    return sim


@router.post("/sessions/{id}/studies")
def start(id: str,body: dict):
    global modal_studies_left
    with lock:
        session = read_session(id,body)
        if session["sim"].request<=0:
            raise HTTPException(422,"Set requested compute above zero")
        cancel_jobs(id)
        key = uuid4().hex
        sim = fresh_study(session["sim"].fork())
        remote = modal_ready() and modal_studies_left > 0
        use_local = not remote and local_model_ready()
        version = checkpoint.parent.name if remote or use_local else "heuristic"
        job = dict(id=key,session_id=id,revision=session["revision"],seed=session["seed"],
                   model_version=version,status="queued",progress="Waiting for worker",solves=0,
                   proposals=[],activity=[],best=None,cancel=Event(),snapshot=sim)
        jobs[key] = job
        if remote:
            modal_studies_left -= 1
        job["inference_backend"] = "Modal" if remote else "local" if use_local else "heuristic"

    def publish(update):
        with lock:
            job.update(update)

    def run():
        try:
            publish(dict(status="running"))
            if remote:
                import modal
                from modal.exception import TimeoutError as ModalTimeout
                fn = modal.Function.from_name(modal_app,"infer")
                call = fn.spawn(version,simulator=sim,stop_at=modal_until)
                while not job["cancel"].is_set():
                    try:
                        publish(call.get(timeout=1))
                        break
                    except (TimeoutError, ModalTimeout):
                        continue
                else:
                    call.cancel()
                    raise InterruptedError()
            else:
                study(sim,local_model() if use_local else None,publish,job["cancel"].is_set)
        except InterruptedError:
            publish(dict(status="cancelled",progress="Study superseded"))
        except Exception as exc:
            publish(dict(status="error",progress=str(exc) or type(exc).__name__))
    worker.submit(run)
    return dict(id=key,revision=job["revision"])


@router.get("/studies/{id}")
def result(id: str):
    with lock:
        if id not in jobs:
            raise HTTPException(404,"Unknown study")
        return deepcopy({k:v for k,v in jobs[id].items() if k not in ("snapshot","cancel")})


@router.get("/studies/{id}/export")
def export(id: str):
    return result(id)


@router.post("/sessions/{id}/studies/{key}/branch")
def branch(id: str,key: str,body: dict):
    with lock:
        session = read_session(id,body)
        job = jobs.get(key)
        if not job or job["session_id"]!=id:
            raise HTTPException(404,"Unknown study")
        if job["revision"]!=session["revision"]:
            raise HTTPException(409,"Proposal belongs to an earlier revision")
        proposal = next((p for p in job["proposals"] if p["id"]==body.get("proposal")),None)
        if proposal is None:
            raise HTTPException(422,"Choose an available proposal")
        sim = job["snapshot"].fork()
        case = proposal["scenario"]
        sim.install({p["bus"]:p["mw"] for p in proposal["allocation"]},case.get("soc",.5))
        for delay,event in case.get("edits",[]):
            sim.event_schedule.setdefault(sim.tick+delay,[]).append(event)
        return new_session(sim,session["seed"],proposal["policy"])


@router.post("/sessions/{id}/step")
def step(id: str,body: dict):
    with lock:
        session = read_session(id,body)
        sim = session["sim"].fork()
    mode = body.get("mode","heuristic")
    if mode not in ("hold","heuristic","learned","assisted","flex"):
        raise HTTPException(422,"Unknown control mode")
    global modal_steps_left
    if mode in ("learned","assisted"):
        with lock:
            remote = modal_ready() and modal_steps_left > 0
            if remote:
                modal_steps_left -= 1
            elif not local_model_ready():
                raise HTTPException(409,"Model inference is unavailable")
        if remote:
            import modal
            fn = modal.Function.from_name(modal_app,"infer")
            result = fn.remote(checkpoint.parent.name,simulator=sim,stop_at=modal_until,mode="control")
            action = result["action"]
        else:
            from .planner_v2 import model_policy
            action, _ = model_policy(sim,local_model())
        if mode=="assisted":
            options = [action,*candidates(sim)]
            action = min(options,key=lambda a:ordered(sim.fork().step(a)))
    elif mode=="flex":
        sim,_,_ = flex_step(sim)
    else:
        action = sim.default_action(mode)
    if mode!="flex":
        sim.step(action)
    with lock:
        session = read_session(id,body)
        cancel_jobs(id)
        session["sim"],session["mode"] = sim,mode
        session["revision"] += 1
        return publish_frame(session)


@router.get("/runs/current")
def run_status():
    training = checkpoint.parent/"training.json"
    evaluation = checkpoint.parent/"evaluation.json"
    remote = modal_ready() and (modal_studies_left > 0 or modal_steps_left > 0)
    available = remote or local_model_ready()
    return dict(model_version=checkpoint.parent.name,available=available,
                training=json.loads(training.read_text()) if training.exists() else [],
                inference_backend="Modal" if remote else "local" if available else "heuristic",
                evaluation=json.loads(evaluation.read_text()) if evaluation.exists() else None)
