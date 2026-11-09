"""Paired closed-loop dispatch sweep on fresh held-out Grid2Op episodes."""
import json
from pathlib import Path

import modal

app = modal.App("gridshift-model-sweep")
volume = modal.Volume.from_name("gridshift-artifacts", version=2)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements("requirements-modal.txt")
         .env({"GRID2OP_DATA_PATH":"/artifacts/grid2op","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
         .add_local_python_source("gridshift"))

LOADS = (35., 70., 105.)
SCENARIOS = ("intact", "connected_outage", "outage_hidden_demand_low_charge")


@app.function(image=image,gpu=["L4","A10G"],cpu=4,memory=16384,timeout=240,
              max_containers=3,volumes={"/artifacts":volume})
def sweep(shard: int, only_episode: int | None = None):
    import grid2op
    import numpy as np
    import torch

    from gridshift.data import split_map
    from gridshift.model_v2 import load_resource_checkpoint
    from gridshift.planner_v2 import catalogue, learned_allocation, model_policy, replay, site_order
    from gridshift.sim_v2 import DT, TOTAL_STEPS, ResourceSimulator

    torch.set_num_threads(2)
    grid2op.change_local_dir("/artifacts/grid2op")
    volume.reload()
    root = Path("/artifacts/gridshift-v2-wam-001")
    prior = json.loads((root/"evaluation.json").read_text())
    excluded = {row["episode"] for row in prior["episodes"]} | {517}
    pool = sorted(set(split_map()["test"]) - excluded)
    episodes = np.random.default_rng(33017).choice(pool, size=6, replace=False).tolist()
    initial = load_resource_checkpoint(root/"checkpoint.pt","cuda")
    refined = load_resource_checkpoint(root/"refined/checkpoint.pt","cuda")

    def run(base, allocation, case, policy, model):
        sim = base.fork()
        sim.install(allocation,case.get("soc",.5))
        for offset,event in case.get("edits",[]):
            sim.event_schedule.setdefault(sim.tick+offset,[]).append(event)
        totals = dict(completed_mwh=0.,immediate_lost_mwh=0.,deadline_miss_mwh=0.,existing_lost_mwh=0.)
        trace, witnesses, counterfactuals = [], [], []
        for step in range(TOTAL_STEPS):
            last = sim.observation["loading"].copy()
            action, forecast = (sim.default_action(),[]) if model is None else model_policy(
                sim,model,policy_only=policy=="policy_only")
            pre_step = sim.fork() if only_episode is not None and model is not None and any(
                power > 0 for power in action["battery"]) else None
            obs = sim.step(action)
            if pre_step is not None and any(v["kind"]=="generation_limit" for v in obs["violations"]):
                alternative = dict(action, battery=[min(power,0.) for power in action["battery"]])
                alternate_obs = pre_step.step(alternative)
                counterfactuals.append(dict(step=step+1, observed=obs["violations"],
                                            no_discharge=alternate_obs["violations"]))
            for key in ("completed_mwh","immediate_lost_mwh","deadline_miss_mwh"):
                totals[key] += obs[key]
            totals["existing_lost_mwh"] += obs["lost"]*DT
            witnesses.extend(obs["violations"])
            error = None
            if forecast and not (obs["failure"] or obs["rejected"]):
                estimated = np.asarray(forecast[0]["loading"])
                measured = obs["loading"]
                error = dict(wam=float(np.abs(estimated-measured).mean()),
                             persistence=float(np.abs(last-measured).mean()),
                             actual_overload_branches=int((measured>1.0001).sum()),
                             wam_detected=int(((measured>1.0001)&(estimated>1.0001)).sum()),
                             persistence_detected=int(((measured>1.0001)&(last>1.0001)).sum()))
            trace.append(dict(step=step+1,peak_loading=float(obs["loading"].max()),
                              battery_mwh=sum(c["energy"] for c in sim.campuses.values()),
                              backlog_mwh=float(sim.queue.sum()),forecast_error=error,
                              violations=obs["violations"],action=obs["executed_action"]))
        return dict(**totals,passed=not witnesses,accepted_mw=sum(allocation.values()),
                    allocation=allocation,witness=next(iter(witnesses),None),trace=trace,
                    discharge_counterfactuals=counterfactuals)

    rows = []
    for index in range(shard*2,shard*2+2):
        episode = episodes[index]
        if only_episode is not None and episode!=only_episode:
            continue
        for j,load in enumerate(LOADS):
            if only_episode is not None and load!=105.:
                continue
            kind = SCENARIOS[(j+index)%len(SCENARIOS)]
            sim = ResourceSimulator(episode=episode,seed=6000+index,request=load)
            start = min(12+19*index,len(sim.profiles[0])-45)
            for tick in range(start-11,start+1):
                sim.tick = tick
                sim.observe()
            available,_ = catalogue(sim)
            outage = max(available[1:],key=lambda c:sim.observation["loading"][c["edits"][0][1]["target"]])
            case = dict(id=kind)
            if kind!="intact":
                case["edits"] = list(outage["edits"])
            if kind==SCENARIOS[2]:
                bus = int(sim.observation["demand"].argmax())
                case["edits"].append((6,dict(kind="demand",target=bus,value=1.2,hidden=True)))
                case["soc"] = .15
            order = site_order(sim)
            fixed = {order[0]:load*.55,order[1]:load*.45}
            baseline = replay(sim,{},case)
            modes = [
                ("heuristic",fixed,None),
                ("policy_only",fixed,initial),
                ("deployed_wam",fixed,initial),
                ("refined_wam",fixed,refined),
                ("wam_placement",learned_allocation(sim,initial),initial),
            ]
            results = {name:run(sim,allocation,case,name,model) for name,allocation,model in modes}
            rows.append(dict(episode=episode,start_tick=start,requested_mw=load,scenario=case,
                             baseline_passed=baseline["passed"],baseline_witness=baseline["witness"],
                             results=results))
            print(episode,load,kind,"baseline",baseline["passed"],
                  {name:value["passed"] for name,value in results.items()},flush=True)
    return dict(gpu=torch.cuda.get_device_name(),rows=rows)


@app.local_entrypoint()
def main():
    shards = list(sweep.map(range(3)))
    rows = [row for shard in shards for row in shard["rows"]]
    assert len(rows)==18 and len({row["episode"] for row in rows})==6
    eligible = [row for row in rows if row["baseline_passed"]]
    summary = {}
    for mode in rows[0]["results"]:
        modes = [row["results"][mode] for row in eligible]
        forecasts = [step["forecast_error"] for result in modes for step in result["trace"] if step["forecast_error"]]
        summary[mode] = dict(
            passes=sum(r["passed"] for r in modes),
            full_request_passes=sum(r["passed"] and abs(r["accepted_mw"]-row["requested_mw"])<1e-5
                                    for row,r in zip(eligible,modes,strict=True)),
            verified_mw=sum(r["accepted_mw"] for r in modes if r["passed"]),
            forecast_steps=len(forecasts),
            mean_loading_error={k:sum(r[k] for r in forecasts)/len(forecasts) for k in ("wam","persistence")} if forecasts else {},
            overload_branch_counts={k:sum(r[k] for r in forecasts) for k in
                                    ("actual_overload_branches","wam_detected","persistence_detected")} if forecasts else {},
        )
    result = dict(gpus=[s["gpu"] for s in shards], eligible=len(eligible), summary=summary, rows=rows)
    path = Path("artifacts/gridshift-v2-wam-001/model-sweep.json")
    path.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(eligible=len(eligible),summary=summary),indent=2))


@app.local_entrypoint()
def diagnose():
    result = sweep.remote(1,243)
    # The sample order is fixed. Validate it before publishing evidence.
    row, = result["rows"]
    assert row["episode"]==243 and row["scenario"]["id"]=="connected_outage"
    path = Path("artifacts/gridshift-v2-wam-001/model-sweep-discharge.json")
    path.write_text(json.dumps(result,indent=2))
    for mode,value in row["results"].items():
        print(mode,value["discharge_counterfactuals"])
