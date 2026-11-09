"""Track 2 capacity experiment. The AC solver scores every proposal."""
import argparse
import json
from pathlib import Path

import numpy as np

from .sim import HOLD, PRIMARY, Simulator


def score(sim, bus, cap, outages, include_intact=True):
    cases = []
    for line in ([None] if include_intact else []) + list(outages):
        trial = sim.fork()
        trial.bus, trial.request, trial.cap = bus, cap, cap
        if line is not None:
            trial.opened[line] = True
        o = trial.observe(replace=True)
        worst = int(np.argmax(o["loading"]))
        cases.append(dict(outage=line, feasible=not o["event"] and o["served"] >= cap-1e-3, failure=o["failure"], served=o["served"],
                          worst_line=worst, loading=float(o["loading"][worst]), lost=o["lost"],
                          min_voltage=float(o["voltage"][o["voltage"]>0].min(initial=1.)),
                          max_voltage=float(o["voltage"].max(initial=1.))))
    return dict(bus=bus, cap_mw=cap, feasible=all(c["feasible"] for c in cases), cases=cases)


def experiment(sim, model=None):
    sim = sim.fork()
    sim.request = sim.cap = 0.
    base = sim.observe(replace=True)
    busiest = np.argsort(-base["loading"])
    outages = []
    excluded = []
    for i in busiest:
        i = int(i)
        if not base["active"][i] or not sim.legal({"kind":"line","a":i,"value":0.}): continue
        if not score(sim,sim.bus,0.,[i])["feasible"]:
            excluded.append(i)
            continue
        outages.append(i)
        if len(outages)==4: break
    buses = [sim.bus, *[int(i) for i in np.argsort(-base["generation"]) if i!=sim.bus][:2]]
    attempts = []
    best = None
    def propose(bus, cap, source="search"):
        nonlocal best
        result = score(sim,bus,cap,outages)
        if result["feasible"] and (best is None or cap > best["cap_mw"]):
            best = result
        result["source"] = source
        result["incumbent_mw"] = None if best is None else best["cap_mw"]
        attempts.append(result)
        return result["feasible"]
    target = float(round(np.maximum(base["demand"],0).sum()*.1,1))
    for bus in buses:
        if not propose(bus,0.): continue
        low,high=0.,target
        if model is not None:
            from .control import predict
            proposal_state=sim.fork()
            proposal_state.bus=bus
            proposal_state.request=target
            proposal_state.observe(replace=True)
            choice=predict(proposal_state,model,caps_only=True)[0][0]
            proposed=round(target*choice["action"]["value"],2)
            passed=propose(bus,proposed,"wam")
            attempts[-1]["predicted_violations"]=sum(o["event"] for o in choice["trajectory"])
            if passed: low=proposed
            else: high=proposed
            if low==target: continue
        if high==target and propose(bus,target): continue
        for _ in range(7):
            midpoint=round((low+high)/2,2)
            if midpoint in (low,high): break
            if propose(bus,midpoint): low=midpoint
            else: high=midpoint
    return dict(network=sim.name, tick=sim.tick, tested_outages=outages, baseline_infeasible_outages=excluded, requested_mw=target,
                attempts=attempts, best=best,
                method="AC-verified adaptive capacity search on selected single outages",
                scope="One operating point, three candidate buses, selected outages, 10% load search ceiling")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--full",action="store_true")
    p.add_argument("--checkpoint")
    a=p.parse_args()
    sim=Simulator(PRIMARY,0,0,not a.full)
    for _ in range(11):sim.step(HOLD)
    from .control import load_model
    result=experiment(sim,load_model(a.checkpoint))
    Path(a.output).write_text(json.dumps(result,indent=2))
    print({k:result[k] for k in ("tested_outages","requested_mw")})
    print("proposals",len(result["attempts"]),"best MW",None if not result["best"] else result["best"]["cap_mw"])
