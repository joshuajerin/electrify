import json
import os
from pathlib import Path
from threading import RLock, Event
from concurrent.futures import ThreadPoolExecutor
import copy
from uuid import uuid4

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .control import assisted, predict
from .data import disturbance
from .sim import HOLD, PRIMARY, Simulator

torch.set_num_threads(2)
checkpoint = os.getenv("GRIDSHIFT_CHECKPOINT","artifacts/checkpoint.pt")
model = None
model_name = Path(checkpoint).parent.name if model else "AC simulator preview"
smoke = os.getenv("GRIDSHIFT_SMOKE","0") == "1"
app = FastAPI(title="Grid Shift")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173","http://127.0.0.1:5173","http://localhost:4173","http://127.0.0.1:4173"],allow_methods=["*"],allow_headers=["*"])
PRESETS = {
    "senoko": {"name":"Senoko Power Station","region":"Singapore","technology":"Gas combined cycle","demand":1.02},
    "surry": {"name":"Surry Power Station","region":"Virginia, USA","technology":"Nuclear","demand":1.0},
    "drax": {"name":"Drax Power Station","region":"North Yorkshire, England","technology":"Biomass","demand":1.04},
}
sessions = {}
lock = RLock()
study_pool = ThreadPoolExecutor(max_workers=1)
missions = {}


@app.post("/sessions/{id}/missions")
def start_mission(id: str, body: dict):
    from .mission import study, Cancelled
    with lock:
        session = get_session(id, body)
        if session["sim"].request <= 0:
            raise HTTPException(422, "Set requested compute above zero")
        for job in missions.values():
            if job["session"] == id and job["status"] in ("queued", "running"):
                job["cancel"].set()
        key = uuid4().hex
        sim = session["sim"].fork()
        job = dict(id=key, session=id, revision=session["revision"], requested_mw=sim.request,
                   status="queued", stage="queued", progress="Waiting for study worker", proposals=[],
                   cancel=Event(), snapshot=sim, model=model_name)
        missions[key] = job

    def publish(update):
        with lock: job.update(update)

    def work():
        try:
            if job["cancel"].is_set(): raise Cancelled()
            publish(dict(status="running"))
            study(sim, model, publish, job["cancel"].is_set)
        except Cancelled:
            publish(dict(status="cancelled", progress="Superseded by a newer study"))
        except Exception as error:
            publish(dict(status="error", progress=str(error)))
    study_pool.submit(work)
    return dict(id=key, revision=job["revision"])


@app.get("/missions/{key}")
def mission_status(key: str):
    with lock:
        if key not in missions: raise HTTPException(404, "Unknown mission")
        return copy.deepcopy({k:v for k,v in missions[key].items() if k not in ("cancel", "snapshot")})


@app.post("/sessions/{id}/missions/{key}/branch")
def mission_branch(id: str, key: str, body: dict):
    with lock:
        session = get_session(id, body)
        job = missions.get(key)
        if job is None or job["session"] != id: raise HTTPException(404, "Unknown mission")
        if job["revision"] != session["revision"]: raise HTTPException(409, "Study belongs to an earlier grid state")
        index = body.get("proposal")
        if job["status"] != "complete" or type(index) is not int or not 0 <= index < len(job["proposals"]):
            raise HTTPException(422, "Choose a completed proposal")
        proposal = job["proposals"][index]
        sim = job["snapshot"].fork()
        sim.bus, sim.cap = proposal["bus"], proposal["cap_mw"]
        sim.observe(replace=True)
        return new_session(sim)


def get_session(id,body=None):
    if id not in sessions: raise HTTPException(404,"Unknown session")
    state=sessions[id]
    if body is not None and body.get("revision")!=state["revision"]:
        raise HTTPException(409,"Stale state revision")
    return state


def render(session):
    s=session["sim"]; o=s.observation
    preset = getattr(s,"preset",None)
    site = PRESETS.get(preset)
    return dict(preset=preset,site=None if site is None else {**site,"anchor_bus":s.site_anchor,"grid":"synthetic"},
                id=session["id"],revision=session["revision"],tick=s.tick,network=s.name,model=model_name,
                settings=dict(demand_scale=s.demand_scale,hidden_bus=s.hidden_bus,hidden_scale=s.hidden_scale,rating_scale=s.rating_scale.tolist()),
                hidden=s.hidden_bus>=0,action=session.get("action"),solver_intervened=session.get("intervened",False),
                buses=[dict(id=i,voltage=float(o["voltage"][i]),demand=float(o["demand"][i]),
                            generation=float(o["generation"][i]),eligible=True) for i in range(s.n)],
                lines=[dict(id=i,**{"from":int(u),"to":int(v)},loading=float(o["loading"][i]),
                            flow=float(o["flow"][i]),active=bool(o["active"][i]),
                            rating=float(s.meta["ratings"][i]*s.rating_scale[i])) for i,(u,v) in enumerate(s.edges.T)],
                compute=dict(bus=s.bus,requested=s.request,cap=s.cap,served=o["served"]),
                events=[label for active,label in ((o["failure"],"AC solve failed"),(o["event"],"Operating limit violation"),
                        (o.get("illegal"),"Action rejected by operating rules"),(o["lost"]>0.001,"Existing load disconnected")) if active])


def save(session):
    state=render(session)
    session["frames"].append((state,session["sim"].snapshot()))
    return state


def new_session(sim):
    id=uuid4().hex
    session=dict(id=id,sim=sim,revision=0,frames=[])
    sessions[id]=session
    return save(session)


def apply_edit(sim,body):
    k=body.get("kind")
    try:
        v=float(body["value"])
        i=int(body.get("target",0))
        if not np.isfinite(v): raise ValueError()
        if k in ("line","rating") and not 0<=i<len(sim.opened): raise ValueError()
        if k=="demand" and not 0<=i<sim.n: raise ValueError()
        if k=="bus" and (not v.is_integer() or not 0<=v<sim.n): raise ValueError()
        if k=="compute" and not 0<=v<=5000: raise ValueError()
        if k=="rating" and not .1<=v<=2: raise ValueError()
        if k=="demand" and not .1<=v<=3: raise ValueError()
        if k=="line" and v not in (0,1): raise ValueError()
        sim.edit(dict(kind=k,value=v,target=i,hidden=bool(body.get("hidden"))))
    except (KeyError,TypeError,ValueError,IndexError):
        raise HTTPException(422,"Invalid scenario edit") from None


@app.post("/sessions")
def create(body:dict):
    scenario=body.get("scenario","baseline")
    seed=int(body.get("seed",0))
    with lock:
        sim=Simulator(PRIMARY,0,seed,smoke)
        if scenario in PRESETS:
            sim.preset = scenario
            generator = int(np.argmax(sim.meta["gen_pmax"]))
            sim.site_anchor = int(sim.meta["gen_to_subid"][generator])
            sim.edit({"kind":"demand","value":PRESETS[scenario]["demand"]})
        for _ in range(11): sim.step(HOLD)
        if scenario in ("outage","hidden"):
            disturbance(sim,1 if scenario=="outage" else 2,"test",sim.rng,3)
        return new_session(sim)


@app.post("/sessions/{id}/edits")
def edit(id:str,body:dict):
    with lock:
        session=get_session(id,body)
        sim=session["sim"].fork()
        apply_edit(sim,body)
        session["sim"]=sim
        session["revision"]+=1
        return save(session)


@app.post("/sessions/{id}/forecast")
def forecast(id:str,body:dict):
    with lock:
        session=get_session(id,body)
        sim=session["sim"].fork()
        revision=session["revision"]
    preview=body.get("edit")
    if preview: apply_edit(sim,preview)
    candidates,ms=predict(sim,model)
    return dict(revision=revision,candidates=candidates,model=model_name,preview=bool(preview),latency_ms=ms)


@app.post("/sessions/{id}/step")
def step(id:str,body:dict):
    with lock:
        session=get_session(id,body); sim=session["sim"]
        mode=body.get("mode","manual")
        action=body.get("action") or HOLD
        intervention=False
        if mode in ("auto","assisted"):
            options,_=predict(sim,model)
            action=options[0]["action"]
            if mode=="assisted":
                action,count=assisted(sim,options)
                intervention=count>0
        try:
            sim.step(action)
        except (KeyError,ValueError,IndexError,TypeError):
            raise HTTPException(422,"Invalid operator action") from None
        session["action"]=action
        session["intervened"]=intervention
        session["revision"]+=1
        return save(session)


@app.post("/sessions/{id}/fork")
def fork(id:str,body:dict):
    with lock:
        session=get_session(id,body)
        index=body.get("frame")
        if index is None: index=len(session["frames"])-1
        if not isinstance(index,int) or not 0<=index<len(session["frames"]):
            raise HTTPException(422,"Unknown frame")
        return new_session(session["frames"][index][1].fork())


@app.get("/sessions/{id}/timeline")
def timeline(id:str):
    with lock:
        return {"frames":[f[0] for f in get_session(id)["frames"]]}


@app.get("/models/{run_id}/metrics")
def metrics(run_id:str):
    p=Path(checkpoint).parent/"metrics.json"
    if not p.exists(): raise HTTPException(404,"No evaluation for this checkpoint")
    return json.loads(p.read_text())


@app.post("/sessions/{id}/capacity")
def capacity(id:str,body:dict):
    from .judging import experiment
    with lock:
        session=get_session(id,body)
        sim=session["sim"].fork()
        revision=session["revision"]
    return {"revision":revision,**experiment(sim,model)}


from .api_v2 import router as resource_router

app.include_router(resource_router)
