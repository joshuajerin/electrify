"""Authenticated, bounded Modal inference for Grid Shift studies."""
import re
import time

import modal

app = modal.App("gridshift-inference-v2")
volume = modal.Volume.from_name("gridshift-artifacts",version=2)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements("requirements-modal.txt")
         .env({"GRID2OP_DATA_PATH":"/artifacts/grid2op","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
         .add_local_python_source("gridshift"))


@app.function(image=image,gpu=["L4","A10G"],cpu=2,memory=8192,timeout=120,
              max_containers=1,scaledown_window=10,volumes={"/artifacts":volume})
def infer(run_id: str, request_mw: float = 40., simulator=None, stop_at: float | None = None, mode: str = "study"):
    from pathlib import Path
    import grid2op
    import torch
    from gridshift.model_v2 import load_resource_checkpoint
    from gridshift.planner_v2 import learned_allocation, model_policy, study
    from gridshift.sim_v2 import ResourceSimulator
    if not re.fullmatch("[a-zA-Z0-9_-]+",run_id):
        raise ValueError("Invalid checkpoint ID")
    if stop_at is not None and time.time()>stop_at:
        raise ValueError("Modal inference reservation expired")
    start = time.monotonic()
    torch.set_num_threads(2)
    volume.reload()
    root = Path("/artifacts")/run_id
    model = load_resource_checkpoint(root/"checkpoint.pt","cuda")
    result = dict(model_version=run_id,planner_version="flex-repair-v1",
                  inference_backend="Modal",gpu=torch.cuda.get_device_name())
    if simulator is not None and mode=="control":
        action,forecast = model_policy(simulator,model)
        result.update(action=action,forecast=forecast)
    elif simulator is not None:
        def cancelled():
            return time.monotonic()-start>100 or (stop_at is not None and time.time()>stop_at)
        study(simulator,model,result.update,cancelled)
    else:
        grid2op.change_local_dir("/artifacts/grid2op")
        sim = ResourceSimulator(smoke=False,seed=42,request=request_mw)
        for tick in range(1,12):
            sim.tick=tick
            sim.observe()
        allocation = learned_allocation(sim,model)
        if not allocation:
            allocation = {sim.eligible[0]:request_mw}
            result["allocation_source"] = "explicit test allocation"
        else:
            result["allocation_source"] = "WAM"
        sim.install(allocation)
        action,forecast = model_policy(sim,model)
        actual = sim.step(action)
        result.update(allocation=allocation,action=action,forecast=forecast,
                      observed_loading=actual["loading"].tolist(),violations=actual["violations"])
    result["inference_seconds"] = time.monotonic()-start
    return result


@app.local_entrypoint()
def demo(run_id: str = "gridshift-v2-wam-001", request_mw: float = 40.):
    import json
    from pathlib import Path
    result = infer.remote(run_id,request_mw)
    path = Path("artifacts")/run_id/"modal-inference.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    print({k:result[k] for k in ("model_version","gpu","inference_seconds")})
