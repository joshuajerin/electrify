import modal

app = modal.App("gridshift")
volume = modal.Volume.from_name("gridshift-artifacts", create_if_missing=True, version=2)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements("requirements-modal.txt")
         .env({"GRID2OP_DATA_PATH":"/artifacts/grid2op","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
         .add_local_python_source("gridshift"))
mount = {"/artifacts":volume}


@app.function(image=image,cpu=2,memory=8192,timeout=1800,volumes=mount)
def prepare_data(smoke: bool=False):
    from gridshift.sim import PRIMARY, TRANSFER, scenarios
    use_data()
    names = {name:scenarios(name,smoke) for name in (PRIMARY,TRANSFER)}
    volume.commit()
    print({k:len(v) for k,v in names.items()},flush=True)
    return names


@app.local_entrypoint()
def prepare(smoke: bool=False):
    prepare_data.remote(smoke)


def use_data():
    import grid2op
    from pathlib import Path
    Path("/artifacts/grid2op").mkdir(parents=True,exist_ok=True)
    grid2op.change_local_dir("/artifacts/grid2op")


@app.function(image=image,cpu=2,memory=4096,timeout=7200,max_containers=8,volumes=mount)
def generate_worker(run_id,worker,windows,smoke):
    import time
    from gridshift.data import generate
    volume.reload()
    use_data()
    directory=f"/artifacts/{run_id}/data"
    start=time.monotonic()
    chunk=windows if smoke else 100
    completed=0
    for offset in range(0,windows,chunk):
        if time.monotonic()-start > 6600: break
        count=min(chunk,windows-offset)
        generate(directory,worker*1000+offset//chunk,count,smoke)
        completed+=count
        volume.commit()
        print(f"worker={worker} windows={completed}/{windows}",flush=True)
    return completed


@app.function(image=image,gpu=["L4","A10G"],cpu=4,memory=16384,timeout=21600,max_containers=1,volumes=mount)
def train_run(run_id,variant,width,steps,seconds,smoke):
    from gridshift.train import train
    volume.reload()
    return train(f"/artifacts/{run_id}/data",f"/artifacts/{run_id}/{variant}",width,steps,smoke,variant,seconds,volume.commit)


@app.function(image=image,cpu=2,memory=4096,timeout=5400,max_containers=4,volumes=mount)
def evaluate_worker(run_id,index,episodes,smoke):
    from pathlib import Path
    from gridshift.evaluate import evaluate
    volume.reload()
    use_data()
    root=Path("/artifacts")/run_id
    destination=root/f"eval-{index}.json"
    result=evaluate(root/"wam"/"checkpoint.pt",root/"data",destination,episodes,smoke,episodes*index,index==0)
    volume.commit()
    return result


@app.function(image=image,cpu=1,memory=3072,timeout=39000,volumes=mount)
def orchestrate(run_id: str,budget_usd:float,smoke:bool,windows:int):
    import json
    from pathlib import Path
    from gridshift.data import manifest
    from gridshift.evaluate import summarize
    if (smoke and budget_usd < 1) or (not smoke and budget_usd < 25):
        raise ValueError("Use at least $1 for smoke or the authorized $25 for full")
    if budget_usd>25: raise ValueError("Budget exceeds authorization")
    if not 1<=windows<=50000: raise ValueError("Use between 1 and 50,000 windows")
    volume.reload()
    use_data()
    root=Path("/artifacts")/run_id
    root.mkdir(parents=True,exist_ok=True)
    receipt=root/"launch.json"
    if receipt.exists():
        raise ValueError("Run ID already exists. Inspect artifacts before resuming to avoid rebilling.")
    # Resource ceilings are expressed in container hours. Rates include a
    # conservative allowance for the published CPU, memory and L4 charges.
    allocations = {"prepare_cpu_hours": .5, "generation_container_hours": 16,
                   "training_l4_hours": 18200/3600, "evaluation_container_hours": 6,
                   "orchestrator_hours": 39000/3600}
    estimate = .90 if smoke else 21.50
    if estimate > budget_usd: raise ValueError("Estimated ceiling exceeds run budget")
    receipt.write_text(json.dumps(dict(budget_usd=budget_usd,estimated_ceiling_usd=estimate,
                                       limits=allocations,requested_windows=windows,smoke=smoke),indent=2))
    volume.commit()
    prepare_data.remote(smoke)
    volume.reload()
    if not smoke:
        from gridshift.check import check
        print("PARITY",check(False),flush=True)
    workers=1 if smoke else 8
    counts=[windows//workers + int(i<windows%workers) for i in range(workers)]
    completed=sum(generate_worker.starmap((run_id,i,n,smoke) for i,n in enumerate(counts) if n))
    volume.reload()
    m=manifest(root/"data",smoke)
    m["windows"]=completed
    m["future_transitions"]=completed*24
    (root/"data"/"manifest.json").write_text(json.dumps(m,indent=2))
    volume.commit()
    print("DATA",completed,"windows",flush=True)
    if not smoke and completed<2000:
        raise RuntimeError(f"Insufficient data for the full run: {completed} windows")
    width=768
    if not smoke:
        train_run.remote(run_id,"world",128,300,1800,False)
    train_run.remote(run_id,"wam",width,12 if smoke else 10000,180 if smoke else 14000,smoke)
    if not smoke:
        train_run.remote(run_id,"policy_only",768,300,2400,False)
    parts=list(evaluate_worker.starmap((run_id,i,1 if smoke else 25,smoke) for i in range(1 if smoke else 4)))
    volume.reload()
    result=summarize(parts)
    result["dataset_windows"]=completed
    result["parameters"]=51426068 if not smoke else None
    result["smoke"]=smoke
    result["target_windows"]=windows
    (root/"wam"/"metrics.json").write_text(json.dumps(result,indent=2))
    import torch
    checkpoint = torch.load(root/"wam"/"checkpoint.pt",map_location="cpu",weights_only=False)
    torch.save({k:checkpoint[k] for k in ("model","width","step")},root/"wam"/"inference.pt")
    volume.commit()
    return result


@app.local_entrypoint()
def pipeline(run_id: str,budget_usd:float=25,smoke:bool=False,windows:int=50000):
    if not run_id.replace("-","").replace("_","").isalnum():
        raise ValueError("Run ID must contain only letters, digits, hyphens and underscores")
    print(orchestrate.remote(run_id,budget_usd,smoke,6 if smoke else windows))


@app.local_entrypoint()
def export(run_id: str):
    import subprocess
    from pathlib import Path
    target=Path("artifacts")/run_id
    target.mkdir(parents=True,exist_ok=True)
    for remote,local in (("wam/inference.pt","checkpoint.pt"),("wam/metrics.json","metrics.json"),
                         ("wam/training.json","training.json"),("data/manifest.json","manifest.json")):
        subprocess.run(["modal","volume","get","gridshift-artifacts",f"{run_id}/{remote}",str(target/local)],check=True)
