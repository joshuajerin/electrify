"""Paired evaluation on source episodes excluded from training and repair."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import split_map
from .data_v2 import make_window
from .model_v2 import load_resource_checkpoint
from .planner_v2 import catalogue, learned_allocation, model_policy, replay, site_order
from .sim import TRANSFER, scenarios
from .sim_v2 import ResourceSimulator
from .train_v2 import HISTORY
from .train_v2 import train


def fixed_suite(smoke=False, limit=24):
    ids = split_map(smoke=smoke)["test"]
    rng = np.random.default_rng(94721)
    ids = rng.permutation(ids)[:limit]
    suite = []
    for n,episode in enumerate(ids):
        s = ResourceSimulator(episode=int(episode),seed=94721+n,smoke=smoke,request=35.)
        start = min(12+n*17,len(s.profiles[0])-45)
        for tick in range(start-11,start+1):
            s.tick = tick
            s.observe()
        cases,_ = catalogue(s)
        case = cases[(n*7) % len(cases)]
        if n%3==2:
            case = dict(id="derating",soc=.4,edits=[(5,dict(kind="rating",target=int(s.observation["loading"].argmax()),value=.87))])
        baseline = replay(s,{},case)
        suite.append((s,case,baseline["passed"]))
    return suite


def compare(s,case,model,baseline_passed,modes):
    order = site_order(s)
    allocation = {order[0]:s.request*.55,order[1]:s.request*.45} if len(order)>1 else {order[0]:s.request}
    results = {}
    for mode in modes:
        proposal = learned_allocation(s,model) if mode=="learned_allocation" else allocation
        policy = "learned" if mode=="learned_allocation" else mode
        started = time.perf_counter()
        res = replay(s,proposal,case,policy,model)
        row = dict(res["metrics"], accepted_mw=sum(proposal.values()),
                   verified_mw=sum(proposal.values()) if baseline_passed and res["passed"] else 0.,
                   passed=bool(res["passed"]),simulator_calls=res["solves"],seconds=time.perf_counter()-started)
        results[mode] = row
    return results


def summary(rows):
    totals = {}
    for mode in sorted({k for row in rows for k in row["results"]}):
        eligible = [row["results"][mode] for row in rows if row["baseline_passed"] and mode in row["results"]]
        totals[mode] = {"eligible_count":len(eligible)}
        if eligible:
            totals[mode].update({k:float(np.mean([r[k] for r in eligible])) for k in eligible[0]})
    return totals


@torch.inference_mode()
def forecast_error(model, episodes):
    device = next(model.parameters()).device
    values = {i:[] for i in (0,2,5,11)}
    for i,episode in enumerate(episodes):
        row = make_window(episode,15229+i,"test")
        x = {k:torch.from_numpy(v).to(device) for k,v in row.items() if k!="split"}
        state = model.belief([x[k][None] for k in HISTORY],x["links"].long())
        for t in range(12):
            state,(v,e,g,s) = model.imagine_step(state,x["actions"][None,t],x["maps"][None,t],
                                                 x["links"].long(),x["contexts"][None,t])
            if t in values and not row["global_target"][t,4]:
                values[t].append([float((e[0,:,0]-x["edge_target"][t,:,0]).abs().mean()),
                                  float((v[0]-x["voltage"][t]).abs().mean()),
                                  float((x["edge"][-1,:,4]-x["edge_target"][t,:,0]).abs().mean())])
    return {str((t+1)*5):dict(zip(("loading_mae","voltage_mae","persistence_loading_mae"),np.mean(v,axis=0).tolist()),count=len(v))
            for t,v in values.items() if v}


def transfer(model):
    rows = []
    for i in range(min(3,len(scenarios(TRANSFER)))):
        sim = ResourceSimulator(name=TRANSFER,episode=i,seed=442+i,request=10)
        for tick in range(1,12):
            sim.tick=tick
            sim.observe()
        case = catalogue(sim)[0][0]
        baseline = replay(sim,{},case)
        results = compare(sim,case,model,baseline["passed"],("heuristic","learned"))
        rows.append(dict(episode=i,baseline_passed=baseline["passed"],results=results))
    return rows


def paired(before,after):
    old = {r["episode"]:r for r in before if r["baseline_passed"]}
    diffs = []
    for r in after:
        if r["episode"] not in old or not r["baseline_passed"]:
            continue
        diffs.append(r["results"]["learned"]["verified_mw"]-old[r["episode"]]["results"]["learned"]["verified_mw"])
    if not diffs:
        return {"count":0}
    rng = np.random.default_rng(558)
    ci = np.quantile(rng.choice(diffs,(1000,len(diffs)),replace=True).mean(1),[.025,.975])
    return {"count":len(diffs),"mean_verified_mw_delta":float(np.mean(diffs)),"bootstrap_ci95":ci.tolist()}


def evaluate_and_refine(data_dir, initial, output, eval_seconds=220, repair_seconds=100, commit=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    model = load_resource_checkpoint(initial,device)
    out = Path(output); out.mkdir(parents=True,exist_ok=True)
    suite = fixed_suite()
    rows = []
    report = {"suite_size":len(suite),"episodes":rows,"repair_training_windows":0,"status":"initial evaluation"}
    def save():
        report["initial"] = summary(rows)
        (out/"evaluation.json").write_text(json.dumps(report,indent=2))
        if commit: commit()
    started = time.monotonic()
    for s,case,baseline in suite:
        if rows and time.monotonic()-started>=eval_seconds:
            break
        results = compare(s,case,model,baseline,("heuristic","ac","policy_only","learned","learned_allocation"))
        rows.append(dict(episode=s.episode,case=case["id"],baseline_passed=baseline,results=results))
        print({"evaluated":len(rows),"episode":s.episode,"eligible":baseline},flush=True)
        if len(rows)%4==0: save()
    report["forecast"] = forecast_error(model,[r["episode"] for r in rows[:4]])
    report["transfer"] = transfer(model)
    save()

    ids = split_map()["train"]
    failed = []
    for i,episode in enumerate(ids[:10]):
        row = make_window(episode,53017+i,"train",actor=lambda sim:model_policy(sim,model,policy_only=True)[0])
        if row["global_target"][:,3:].any() or row["service"][:,1:3].max()>1e-4:
            failed.append(row)
    report["repair_training_windows"] = len(failed)
    if failed:
        samples = {k:np.stack([row[k] for row in failed]) for k in failed[0]}
        np.savez_compressed(out/"failure-data.npz",**samples)
        del model
        if device=="cuda": torch.cuda.empty_cache()
        refined = train(data_dir,out/"refined",initial,repair_seconds,commit=commit,
                        failure_rows=[(samples,i) for i in range(len(failed))])
        model = load_resource_checkpoint(refined,device)
        after = []
        report["refined_episodes"] = after
        for s,case,baseline in suite[:len(rows)]:
            results = compare(s,case,model,baseline,("learned","learned_allocation"))
            after.append(dict(episode=s.episode,case=case["id"],baseline_passed=baseline,results=results))
            report["refined"] = summary(after)
            report["paired"] = paired(rows,after)
            if len(after)%4==0: save()
    report["status"] = "complete"
    save()
    return {"episodes":len(rows),"failure_windows":len(failed),"status":report["status"]}
