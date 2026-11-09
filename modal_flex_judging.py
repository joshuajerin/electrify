"""Paired AC experiments for flexible compute under temporary thermal limits."""
import json
from pathlib import Path

import modal

app = modal.App("gridshift-flex-judging")
volume = modal.Volume.from_name("gridshift-artifacts", version=2)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements("requirements-modal.txt")
         .env({"GRID2OP_DATA_PATH":"/artifacts/grid2op","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
         .add_local_python_source("gridshift"))


@app.function(image=image,cpu=2,memory=8192,timeout=180,max_containers=2,volumes={"/artifacts":volume})
def evaluate(shard: int):
    import grid2op
    import numpy as np

    from gridshift.data import split_map
    from gridshift.planner_v2 import replay, site_order, study
    from gridshift.sim_v2 import ResourceSimulator

    grid2op.change_local_dir("/artifacts/grid2op")
    volume.reload()
    root = Path("/artifacts/gridshift-v2-wam-001")
    prior = json.loads((root/"evaluation.json").read_text())
    excluded = {r["episode"] for r in prior["episodes"]} | {77,283,243,136,354,510}
    episodes = np.random.default_rng(12501).choice(sorted(set(split_map()["test"])-excluded),12,replace=False)
    results = []
    for episode in episodes[shard*6:shard*6+6]:
        sim = ResourceSimulator(episode=int(episode),seed=7241,request=105.)
        for tick in range(1,13):
            sim.tick = tick
            sim.observe()
        bus = site_order(sim)[0]
        sim.eligible = (bus,)
        allocation = {bus:sim.request}
        intact = replay(sim,allocation,policy="hold")
        baseline = replay(sim,{},policy="hold")
        if not intact["passed"] or not baseline["passed"]:
            results.append(dict(episode=int(episode),skip="Intact reference infeasible"))
            continue
        # Construct a disclosed stress fixture from the reference flows. Use only
        # the fixed zero-load and immediate-processing traces to set the rating.
        start,stop = 8,11
        existing_peak = np.max([f["loading"] for f in baseline["trajectory"][start-1:stop-1]],axis=0)
        full_min = np.min([f["loading"] for f in intact["trajectory"][start-1:stop-1]],axis=0)
        candidates = np.flatnonzero((full_min-existing_peak>.04)&(full_min>.45))
        if not len(candidates):
            results.append(dict(episode=int(episode),skip="No suitable transient thermal window"))
            continue
        line = int(candidates[np.argmax(full_min[candidates]-existing_peak[candidates])])
        rating = float(existing_peak[line]+.8*(full_min[line]-existing_peak[line]))
        transient = dict(id="temporary-thermal-constraint",edits=[
            (start,dict(kind="rating",target=line,value=rating)),
            (stop,dict(kind="rating",target=line,value=1.))])
        permanent = dict(id="sustained-thermal-constraint",edits=transient["edits"][:1])
        report = {}
        study(sim,None,report.update,lambda:False,cases=[dict(id="intact"),transient])
        rigid = next(p for p in report["proposals"] if p["policy"]=="hold")
        flex = next((p for p in report["proposals"] if p["policy"]=="flex"),None)
        sustained_base = replay(sim,{},permanent,policy="hold")
        sustained = replay(sim,allocation,permanent,policy="flex")
        row = dict(episode=int(episode),requested_mw=sim.request,bus=bus,line=line,rating=rating,
                   studied=report,rigid_pass=rigid["passed"],flex_pass=bool(flex and flex["passed"]),
                   sustained=dict(baseline_passed=sustained_base["passed"],passed=sustained["passed"],
                                  witness=sustained["witness"],metrics=sustained["metrics"]))
        results.append(row)
        print(int(episode),line,rigid["passed"],row["flex_pass"],row["sustained"]["passed"],flush=True)
    return results


@app.local_entrypoint()
def main():
    rows = [row for shard in evaluate.map(range(2)) for row in shard]
    paired = [r for r in rows if "studied" in r]
    result = dict(rows=rows,paired=len(paired),fixed_passes=sum(r["rigid_pass"] for r in paired),
                  flex_passes=sum(r["flex_pass"] for r in paired))
    path = Path("artifacts/gridshift-v2-wam-001/flexible-workload-study.json")
    path.write_text(json.dumps(result,indent=2))
    print({k:v for k,v in result.items() if k!="rows"})


@app.function(image=image,cpu=2,memory=8192,timeout=120,volumes={"/artifacts":volume})
def capacity_bounds(episode: int, bus: int, case: dict):
    import grid2op
    from gridshift.planner_v2 import replay, study
    from gridshift.sim_v2 import ResourceSimulator

    grid2op.change_local_dir("/artifacts/grid2op")
    volume.reload()
    sim = ResourceSimulator(episode=episode,seed=7241,request=105.)
    for tick in range(1,13):
        sim.tick = tick
        sim.observe()
    sim.eligible = (bus,)
    comparisons = []
    for load in (80.,85.,90.,95.,100.,105.):
        result = replay(sim,{bus:load},case,policy="hold")
        comparisons.append(dict(mw=load,passed=result["passed"],witness=result["witness"]))
    flexible = replay(sim,{bus:105.},case,policy="flex")
    report = {}
    study(sim,None,report.update,lambda:False,cases=[dict(id="intact"),case])
    repair = report["best"]
    assert repair["policy"]=="flex" and repair["case"]==case["id"]
    assert repair["parent_id"]==report["proposals"][0]["id"] and repair["passed"]
    return dict(episode=episode,bus=bus,scenario=case,fixed=comparisons,
                flexible=dict(mw=105.,passed=flexible["passed"],metrics=flexible["metrics"],
                              peak_backlog_mwh=max(sum(c["backlog_mwh"] for c in frame["campuses"])
                                                   for frame in flexible["trajectory"])),
                repair_grade=repair["grade"], replay_case=repair["case"])


@app.local_entrypoint()
def bound():
    root = Path("artifacts/gridshift-v2-wam-001")
    study = json.loads((root/"flexible-workload-study.json").read_text())
    row = next(row for row in study["rows"] if row.get("flex_pass"))
    reference = next(p for p in row["studied"]["proposals"] if p["policy"]=="hold")
    result = capacity_bounds.remote(row["episode"],row["bus"],reference["scenario"])
    (root/"flexible-capacity-bound.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
