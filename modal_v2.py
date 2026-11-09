"""Bounded Grid Shift training jobs. Run with modal run --detach."""
import modal

volume = modal.Volume.from_name("gridshift-artifacts", version=2)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements("requirements-modal.txt")
         .env({"GRID2OP_DATA_PATH":"/artifacts/grid2op","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
         .add_local_python_source("gridshift"))

app = modal.App("gridshift-v2")
mount = {"/artifacts": volume}


@app.function(image=image, gpu=["L4", "A10G"], cpu=4, memory=16384, timeout=780,
              max_containers=1, volumes=mount)
def electrical(run_id: str, seconds: int, prior_usd: float):
    import json
    import time
    from pathlib import Path
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from gridshift.model import WAM
    from gridshift.train import batch, decode, loss_for

    # Reserve the entire container timeout at the more expensive fallback rate.
    upper = 780 * (.000306 + 4 * .0000131 + 16 * .00000222)
    if not 60 <= seconds <= 650 or not 0 <= prior_usd <= 23 - upper:
        raise ValueError("Run exceeds duration or cumulative budget")
    torch.manual_seed(51)
    torch.set_num_threads(2)
    rng = np.random.default_rng(51)
    started = time.monotonic()
    volume.reload()
    root = Path("/artifacts") / run_id
    root.mkdir(exist_ok=False)
    files = sorted(Path("/artifacts/gridshift-003/data").glob("shard-*.parquet"))
    pool, val = [], []
    for i in np.linspace(0, len(files)-1, min(len(files), 16), dtype=int):
        for row in pq.read_table(files[i]).to_pylist():
            target = pool if row["split"] == "train" else val if row["split"] == "val" else None
            if target is not None:
                target.append(row)
    if not pool or not val:
        raise RuntimeError("No compatible train and validation rows")
    train_sources = {r["source_week"] for r in pool}
    assert train_sources.isdisjoint(r["source_week"] for r in val)
    model = WAM(768).cuda()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n >= 50_000_000
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    ledger = dict(prior_usd=prior_usd, reservation_usd=upper, reserve_usd=2,
                  dataset_shards=len(files), training_sources=len(train_sources), parameters=n,
                  stage="electrical", gpu=torch.cuda.get_device_name())
    (root/"manifest.json").write_text(json.dumps(ledger, indent=2))
    volume.commit()
    print(json.dumps(ledger), flush=True)

    def sample(rows, size):
        chosen = []
        seen = set()
        for idx in rng.choice(len(rows), min(len(rows), 128), replace=False):
            row = rows[idx]
            if row["source_week"] not in seen:
                chosen.append(decode(row))
                seen.add(row["source_week"])
            if len(chosen) == size:
                break
        return batch(chosen, "cuda")

    validation = sample(val, 4)
    def validate():
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            metrics = loss_for(model, validation, "world")[1]
        model.train()
        return metrics

    history = [{"step": 0, "validation": validate()}]
    step = 0
    saved = 0.
    while time.monotonic()-started < seconds or step < 2:
        elapsed = time.monotonic()-started
        stage = "one" if elapsed < seconds*.24 else "world"
        opt.zero_grad(set_to_none=True)
        totals = []
        for _ in range(4):
            x = sample(pool, 2)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = loss_for(model, x, stage)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            (loss/4).backward()
            totals.append(metrics["world"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        step += 1
        elapsed = time.monotonic()-started
        if step == 1 or step % 20 == 0 or elapsed >= seconds:
            log = dict(step=step, elapsed_s=round(elapsed), stage=stage, loss=float(np.mean(totals)),
                       validation=validate())
            history.append(log)
            print(json.dumps(log), flush=True)
        if step == 1 or elapsed - saved >= 110 or elapsed >= seconds:
            tmp = root/"checkpoint.tmp"
            torch.save(dict(model=model.state_dict(), width=768, step=step), tmp)
            tmp.replace(root/"checkpoint.pt")
            torch.save(dict(optimizer=opt.state_dict(), sampler=rng.bit_generator.state,
                            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                            stage=stage, step=step), root/"resume.pt")
            (root/"training.json").write_text(json.dumps(history, indent=2))
            volume.commit()
            saved = elapsed
    return history[-1]


@app.local_entrypoint()
def pretrain(run_id: str = "gridshift-v2-electrical-001", seconds: int = 480, prior_usd: float = 1.5):
    print(electrical.remote(run_id, seconds, prior_usd))

data_volume = modal.Volume.from_name("gridshift-v2-data",create_if_missing=True,version=2)


@app.function(image=image,cpu=2,memory=4096,timeout=660,max_containers=6,
              volumes={"/artifacts":volume,"/data":data_volume})
def generate_resource_worker(run_id: str, worker: int, chunks: int):
    import time
    import grid2op
    from gridshift.data_v2 import generate_shard
    started = time.monotonic()
    grid2op.change_local_dir("/artifacts/grid2op")
    for offset in range(chunks):
        if time.monotonic()-started>500:
            break
        path = generate_shard("/data/"+run_id,worker*100+offset)
        data_volume.commit()
        print(path,flush=True)


@app.function(image=image,cpu=.25,memory=256,timeout=720)
def generation(run_id: str, prior_usd: float, chunks: int):
    reserve = 6*660*(2*.0000131+4*.00000222)+720*(.25*.0000131+.25*.00000222)
    if not 0 <= prior_usd <= 23-reserve or not 1 <= chunks <= 22:
        raise ValueError("Insufficient budget or invalid chunk count")
    print({"prior_usd":prior_usd,"reservation_usd":reserve,"worker_count":6},flush=True)
    list(generate_resource_worker.starmap((run_id,i,chunks) for i in range(6)))


@app.local_entrypoint()
def generate(run_id: str = "gridshift-v2-resource-001", prior_usd: float = 2., chunks: int = 8):
    generation.remote(run_id,prior_usd,chunks)


@app.function(image=image,gpu=["L4","A10G"],cpu=4,memory=16384,timeout=990,max_containers=1,
              volumes={"/artifacts":volume,"/data":data_volume})
def fit_resource(run_id: str, data_id: str, warm_id: str, seconds: int, prior_usd: float):
    from pathlib import Path
    from gridshift.train_v2 import train
    reserve = 990*(.000306+4*.0000131+16*.00000222)
    if not 0<=prior_usd<=23-reserve or not 45<=seconds<=850:
        raise ValueError("Resource fit exceeds spend or time bounds")
    volume.reload()
    data_volume.reload()
    path = Path("/data")/data_id
    if len(list(path.glob("shard-*.json")))<8:
        raise ValueError("Wait for eight committed resource shards")
    print({"prior_usd":prior_usd,"reservation_usd":reserve},flush=True)
    return train(path,Path("/artifacts")/run_id,Path("/artifacts")/warm_id/"checkpoint.pt",
                 seconds,data_volume.reload,volume.commit)


@app.local_entrypoint()
def fit(run_id: str = "gridshift-v2-wam-001", data_id: str = "gridshift-v2-resource-001",
        warm_id: str = "gridshift-v2-electrical-001", seconds: int = 510, prior_usd: float = 3.):
    print(fit_resource.remote(run_id,data_id,warm_id,seconds,prior_usd))


@app.function(image=image,gpu=["L4","A10G"],cpu=4,memory=16384,timeout=600,max_containers=1,
              volumes={"/artifacts":volume,"/data":data_volume})
def evaluate_resource(run_id: str, data_id: str, prior_usd: float):
    from pathlib import Path
    import grid2op
    from gridshift.eval_v2 import evaluate_and_refine
    reserve = 600*(.000306+4*.0000131+16*.00000222)
    if not 0<=prior_usd<=23-reserve:
        raise ValueError("Evaluation exceeds spend bound")
    grid2op.change_local_dir("/artifacts/grid2op")
    volume.reload()
    data_volume.reload()
    root = Path("/artifacts")/run_id
    return evaluate_and_refine(Path("/data")/data_id,root/"checkpoint.pt",root,
                               eval_seconds=155,repair_seconds=75,commit=volume.commit)


@app.local_entrypoint()
def evaluate(run_id: str = "gridshift-v2-wam-001", data_id: str = "gridshift-v2-resource-001",
             prior_usd: float = 4.):
    print(evaluate_resource.remote(run_id,data_id,prior_usd))
