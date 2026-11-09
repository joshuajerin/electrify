import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .control import assisted, load_model, predict
from .data import disturbance, initial, ordered, split_map
from .train import read_rows
from .sim import HOLD, PRIMARY, TRANSFER, scenarios


def forecast_metrics(model, data):
    errors={}
    sample_count=0
    for row,x in read_rows(data):
        if row["split"]!="test": continue
        if sample_count>=1024: break
        sample_count+=1
        n,e,g,a,links=[torch.as_tensor(x[k]) for k in ("node","edge","glob","actions","links")]
        with torch.inference_mode():
            s=model.encode(n[None],e[None],g[None],links)
            v,p,q=model.rollout(s,a[None],links)
        actual=x["edge_target"][...,0]
        for t in (0,2,5):
            if x["global_target"][t,4]: continue
            val=errors.setdefault(t,[])
            predicted=p[0,t,:,0].numpy()
            flags=predicted>1; truth=actual[t]>1
            val.append([np.mean(abs(predicted-actual[t])),np.mean(abs(x["edge"][-1,:,4]-actual[t])),
                        np.mean(abs(v[0,t].numpy()-x["voltage"][t])),
                        (flags&truth).sum(),(flags&~truth).sum(),(~flags&truth).sum()])
    result={}
    for t,rows in errors.items():
        avg=np.mean(rows,0); totals=np.sum(rows,0)
        result[str((t+1)*5)]={"loading_mae":float(avg[0]),"persistence_mae":float(avg[1]),"voltage_mae":float(avg[2]),
            "overload_precision":float(totals[3]/max(1,totals[3]+totals[4])),
            "overload_recall":float(totals[3]/max(1,totals[3]+totals[5]))}
    return result


def ac_lookahead(sim,candidates):
    forecast=sim.causal_fork()
    def evaluate(plan):
        trial=forecast.fork()
        rank=np.zeros(5)
        for a in plan+[HOLD]*(6-len(plan)):
            rank+=ordered(trial.step(a),a)
        return tuple(rank)
    return min(candidates,key=lambda p:evaluate(p["plan"]))["action"]


def run_episode(sim,model,mode,policy_only=False,world_only=False):
    totals=dict(compute_mwh=0.,existing_mwh_lost=0.,violation_intervals=0,solver_interventions=0,switches=0,failures=0,latency_ms=[])
    for _ in range(12):
        if mode=="hold": action=HOLD
        elif mode=="greedy": action=sim.greedy()
        else:
            candidates,ms=predict(sim,model,policy_only=policy_only,world_only=world_only)
            start=time.perf_counter()
            action=candidates[0]["action"]
            if mode=="assisted":
                action,count=assisted(sim,candidates)
                totals["solver_interventions"]+=int(count>0)
            elif mode=="ac": action=ac_lookahead(sim,candidates)
            totals["latency_ms"].append(ms+(time.perf_counter()-start)*1000)
        o=sim.step(action)
        totals["compute_mwh"]+=o["served"]/12
        totals["existing_mwh_lost"]+=o["lost"]/12
        totals["violation_intervals"]+=int(o["event"])
        totals["failures"]+=int(o["failure"])
        totals["switches"]+=int(action["kind"]=="line" and not o["illegal"])
    return totals


def adaptation(sim,model):
    values=[]
    for i in range(7):
        actual=sim.fork()
        target=np.stack([actual.step(HOLD)["loading"] for _ in range(6)])
        if actual.observation["failure"]: break
        forecasts=[predict(sim,model,memory=m)[0] for m in (True,False)]
        result=[]
        for candidates in forecasts:
            hold=next(x for x in candidates if all(a["kind"]=="hold" for a in x["plan"]))
            result.append(float(np.mean(np.abs(np.array([o["loading"] for o in hold["trajectory"]])-target))))
        values.append(result)
        sim.step(HOLD)
    return values


def evaluate(checkpoint,data,output,episodes=100,smoke=False,offset=0,transfer=True):
    torch.set_num_threads(2)
    model=load_model(checkpoint)
    if model is None: raise ValueError("Checkpoint is required")
    ids=split_map(smoke=smoke)["test"]
    if not ids: return {"smoke":True,"reason":"packaged sandbox has no held-out source week"}
    results=[]
    root=Path(checkpoint).parent.parent
    variants={name:load_model(root/name/"checkpoint.pt") for name in ("world","policy_only")}
    for i in range(offset,offset+episodes):
        rng=np.random.default_rng(70001+i)
        sim=initial(PRIMARY,ids[i%len(ids)],70001+i,smoke)
        disturbance(sim,1+i%3,"test",rng)
        row={m:run_episode(sim.fork(),model,m) for m in ("hold","greedy","raw","assisted","ac")}
        for name,variant in variants.items():
            if variant is not None:
                row[name]=run_episode(sim.fork(),variant,"raw",policy_only=name=="policy_only",world_only=name=="world")
        if i%3==1:
            branch=sim.fork()
            branch.step(HOLD)
            row["adaptation"]=adaptation(branch,model)
        results.append(row)
    extra=[]
    if transfer and not smoke:
        count=len(scenarios(TRANSFER,False))
        for i in range(min(12,episodes)):
            sim=initial(TRANSFER,i%count,80000+i,False)
            disturbance(sim,1,"test",np.random.default_rng(i))
            extra.append({m:run_episode(sim.fork(),model,m) for m in ("hold","greedy","raw")})
    payload={"episodes":results,"transfer":extra,"forecast":forecast_metrics(model,data) if offset==0 else {}}
    Path(output).write_text(json.dumps(payload,indent=2))
    return payload


def summarize(parts):
    rows=[r for p in parts for r in p.get("episodes",[])]
    def metrics(rows):
        out={}
        for mode in ("hold","greedy","raw","assisted","ac","world","policy_only"):
            vals=[r[mode] for r in rows if mode in r]
            if not vals: continue
            out[mode]={k:float(np.mean([r[k] for r in vals])) for k in vals[0] if k!="latency_ms"}
            latency=[v for row in vals for v in row["latency_ms"]]
            if latency: out[mode]["latency_p95_ms"]=float(np.percentile(latency,95))
        return out
    pairs=[r["adaptation"] for r in rows if len(r.get("adaptation",[]))==7]
    result={"episode_count":len(rows),"control":metrics(rows),"transfer":metrics([r for p in parts for r in p.get("transfer",[])]),
            "forecast":next((p["forecast"] for p in parts if p.get("forecast")),{}),
            "adaptation_curve":np.mean(pairs,0).tolist() if pairs else []}
    if rows:
        d=np.array([r["raw"]["compute_mwh"]-r["greedy"]["compute_mwh"] for r in rows])
        bootstrap=np.random.default_rng(51).choice(d,(2000,len(d))).mean(1)
        result["raw_minus_greedy_compute_mwh_ci95"]=np.percentile(bootstrap,[2.5,97.5]).tolist()
    return result
