"""Reproduce paired stress replays on Modal without changing deployed weights."""
from pathlib import Path
import json

import modal

app = modal.App("gridshift-edge-cases")
volume = modal.Volume.from_name("gridshift-artifacts", version=2)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements("requirements-modal.txt")
         .env({"GRID2OP_DATA_PATH":"/artifacts/grid2op","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
         .add_local_python_source("gridshift"))


@app.function(image=image,gpu=["L4","A10G"],cpu=4,memory=16384,timeout=240,
              max_containers=1,volumes={"/artifacts":volume})
def check():
    import grid2op
    import numpy as np
    import torch

    from gridshift.data import split_map
    from gridshift.model_v2 import load_resource_checkpoint
    from gridshift.planner_v2 import catalogue, model_policy, replay, site_order
    from gridshift.sim_v2 import DT, TOTAL_STEPS, ResourceSimulator

    torch.set_num_threads(2)
    grid2op.change_local_dir("/artifacts/grid2op")
    root = Path("/artifacts/gridshift-v2-wam-001")
    volume.reload()
    episode = 517
    assert episode in split_map()["test"]
    sim = ResourceSimulator(episode=episode,seed=94721,request=35.)
    for tick in range(1,13):
        sim.tick = tick
        sim.observe()
    eligible = site_order(sim)
    allocation = {eligible[0]:19.25,eligible[1]:15.75}
    biggest_load = int(sim.observation["demand"].argmax())
    cases, islands = catalogue(sim)
    branch = max(cases[1:],key=lambda c:sim.observation["loading"][c["edits"][0][1]["target"]])
    stressed = [
        {"id":"intact"},
        {"id":"battery_at_minimum","soc":.1},
        {"id":"forced_deferral","deferral_steps":6},
        {"id":"hidden_demand_plus_20_percent","edits":[(4,dict(kind="demand",target=biggest_load,value=1.2,hidden=True))]},
        {**branch,"id":"loaded_connected_branch_outage"},
        {"id":"battery_unavailable","disabled_batteries":list(allocation)},
    ]

    def evaluate(case,model,mode):
        s = sim.fork()
        s.install(allocation,case.get("soc",.5))
        s.disabled_batteries.update(case.get("disabled_batteries",[]))
        s.observe(replace=True)
        for offset,event in case.get("edits",[]):
            s.event_schedule.setdefault(s.tick+offset,[]).append(event)
        totals = dict(completed_mwh=0.,immediate_lost_mwh=0.,deadline_miss_mwh=0.,existing_lost_mwh=0.)
        trace,violations = [],[]
        for i in range(TOTAL_STEPS):
            before = s.observation["loading"].copy()
            if mode=="heuristic":
                action,forecast = s.default_action(),[]
            else:
                action,forecast = model_policy(s,model,policy_only=(mode=="policy_only"))
            if i<case.get("deferral_steps",0):
                action = dict(action,processing=[mw*.56 for mw in allocation_in_sim_order])
                # The external clamp changes the action the forecast was conditioned on.
                forecast = []
            o = s.step(action)
            for k in ("completed_mwh","immediate_lost_mwh","deadline_miss_mwh"):
                totals[k] += o[k]
            totals["existing_lost_mwh"] += o["lost"]*DT
            violations.extend(o["violations"])
            compare = None if o["failure"] or not forecast or o["rejected"] else dict(
                wam=float(np.abs(np.asarray(forecast[0]["loading"])-o["loading"]).mean()),
                persistence=float(np.abs(before-o["loading"]).mean()))
            trace.append(dict(step=i+1,peak_loading=float(o["loading"].max()),
                              backlog_mwh=float(s.queue.sum()),battery_mwh=sum(c["energy"] for c in s.campuses.values()),
                              battery_commands=action["battery"],executed=o["executed_action"],
                              masked_bus=s.mask_bus,loading_error=compare,violations=o["violations"]))
        return dict(**totals,passed=not violations,witness=next(iter(violations),None),
                    peak_backlog_mwh=max(f["backlog_mwh"] for f in trace),
                    terminal_backlog_mwh=float(s.queue.sum()),
                    terminal_battery_mwh=trace[-1]["battery_mwh"],trace=trace)

    allocation_in_sim_order = [allocation[b] for b in sorted(allocation)]
    report = dict(episode=episode,allocation=allocation,gpu=torch.cuda.get_device_name(),
                  excluded_islanding_outages=islands,cases=[])
    for case in stressed:
        baseline = replay(sim,{},case)
        report["cases"].append(dict(scenario=case,baseline_passed=baseline["passed"],
                      baseline_witness=baseline["witness"],results={"heuristic":evaluate(case,None,"heuristic")}))
    for name,path,mode in (
        ("initial_wam_reserve",root/"checkpoint.pt","learned"),
        ("initial_policy_only",root/"checkpoint.pt","policy_only"),
        ("refined_wam_reserve",root/"refined/checkpoint.pt","learned"),
    ):
        model = load_resource_checkpoint(path,"cuda")
        for row in report["cases"]:
            row["results"][name] = evaluate(row["scenario"],model,mode)
            print(name,row["scenario"]["id"],row["results"][name]["passed"],flush=True)
        del model
    return report


@app.local_entrypoint()
def main():
    report = check.remote()
    out = Path("artifacts/gridshift-v2-wam-001/edge-cases-reserve.json")
    out.write_text(json.dumps(report,indent=2))
    for row in report["cases"]:
        print(row["scenario"]["id"],row["baseline_passed"],{
            k:(v["passed"],v["witness"]["kind"] if v["witness"] else None)
            for k,v in row["results"].items()})


@app.function(image=image,gpu=["L4","A10G"],cpu=4,memory=16384,timeout=240,
              max_containers=1,volumes={"/artifacts":volume})
def check_held_out():
    import grid2op
    import torch

    from gridshift.eval_v2 import compare, fixed_suite, summary
    from gridshift.model_v2 import load_resource_checkpoint

    torch.set_num_threads(2)
    grid2op.change_local_dir("/artifacts/grid2op")
    volume.reload()
    root = Path("/artifacts/gridshift-v2-wam-001")
    previous = json.loads((root/"evaluation.json").read_text())
    suite = fixed_suite(limit=len(previous["episodes"]))
    rows = []
    for before,(sim,case,baseline) in zip(previous["episodes"],suite,strict=True):
        assert (sim.episode,case["id"],baseline) == (before["episode"],before["case"],before["baseline_passed"])
        rows.append(dict(episode=sim.episode,case=case["id"],baseline_passed=baseline,results={}))
    for name,path in (("initial_wam_reserve",root/"checkpoint.pt"),
                      ("refined_wam_reserve",root/"refined/checkpoint.pt")):
        model = load_resource_checkpoint(path,"cuda")
        for row,(sim,case,baseline) in zip(rows,suite,strict=True):
            row["results"][name] = compare(sim,case,model,baseline,("learned",))["learned"]
            print(name,sim.episode,baseline,row["results"][name]["passed"],flush=True)
        del model
    return dict(episodes=rows, summary=summary(rows), previous=previous["initial"],
                previous_refined=previous["refined"], gpu=torch.cuda.get_device_name())


@app.local_entrypoint()
def held_out():
    report = check_held_out.remote()
    out = Path("artifacts/gridshift-v2-wam-001/held-out-reserve.json")
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report["summary"],indent=2))
