import argparse
import io
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from .model import WAM


def read_rows(path):
    for file in sorted(Path(path).glob("shard-*.parquet")):
        for row in pq.read_table(file).to_pylist():
            with np.load(io.BytesIO(row.pop("payload"))) as blob:
                yield row, dict(blob)


@lru_cache(maxsize=6)
def shard_rows(file):
    return pq.read_table(file).to_pylist()


def decode(row):
    with np.load(io.BytesIO(row["payload"])) as blob:
        return dict(blob)


def batch(rows, device):
    return {k: torch.as_tensor(np.stack([x[k] for x in rows]),device=device) for k in rows[0]}


def loss_for(model, x, stage):
    links = x["links"][0]
    state = model.encode(x["node"],x["edge"],x["glob"],links)
    horizon = 1 if stage == "one" else 6
    v,e,g = model.rollout(state,x["actions"][:,:horizon],links)
    y = x["global_target"][:,:horizon]
    solved = 1-y[...,4]
    def regression(pred, target, mask):
        errors = F.smooth_l1_loss(pred,target,reduction="none")*mask
        return errors.sum()/mask.expand_as(errors).sum().clamp_min(1)
    voltage = regression(v,x["voltage"][:,:horizon],solved[...,None])
    loading = regression(e[...,0],x["edge_target"][:,:horizon,:,0],solved[...,None])
    flow = regression(e[...,1:],x["edge_target"][:,:horizon,:,1:],solved[...,None,None])
    target = y[:,:,None,:].expand_as(g)
    served = regression(g[...,:3],target[...,:3],solved[...,None,None])
    events = F.binary_cross_entropy_with_logits(g[...,3:],target[...,3:],pos_weight=g.new_tensor([1.,3.]))
    policy = v.new_zeros(())
    if stage in ("policy","joint","policy_only"):
        losses = []
        for i in range(3):
            labels = x["labels"][:,i]
            active = labels >= 0
            if not active.any(): continue
            options = x["options"][:,i]
            logits = model.logits(state,options,x["actions"][:,:i] if i else None)
            valid = options[...,0] != 0
            valid[:,0] = True
            logits = logits.masked_fill(~valid,-1e4)
            losses.append(F.cross_entropy(logits[active],labels[active]))
        if losses: policy = torch.stack(losses).mean()
    world = voltage+loading+.5*flow+served+2*events
    total = policy if stage in ("policy","policy_only") else world+policy
    return total, {"world":float(world.detach()),"policy":float(policy.detach()),
                   "loading_mae":float(((e[...,0]-x["edge_target"][:,:horizon,:,0]).abs()*solved[...,None]).sum().detach()/((solved.sum()*e.shape[-2]).clamp_min(1)))}


def train(path,out,width=128,steps=100,smoke=False,variant="wam",seconds=600,commit=None):
    torch.manual_seed(51)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files=sorted(Path(path).glob("shard-*.parquet"))
    if not files: raise ValueError("No training shards")
    val=[]
    if not smoke:
        for file in files[:4]:
            val.extend(decode(row) for row in shard_rows(file) if row["split"]=="val")
            if len(val)>=16: break
    model = WAM(width).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    rng = np.random.default_rng(51)
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    latest = out/"checkpoint.pt"
    start = 0
    history = []
    if latest.exists():
        ckpt = torch.load(latest,map_location="cpu",weights_only=False)
        model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["optimizer"])
        rng.bit_generator.state=ckpt["rng"]; start=ckpt["step"]; history=ckpt["history"]
        torch.set_rng_state(ckpt["torch_rng"])
        if device=="cuda" and ckpt.get("cuda_rng"): torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    stop = time.monotonic()+seconds
    last_save = time.monotonic()
    if smoke:
        expert_rows = [decode(row) for file in files for row in shard_rows(file)
                       if row["split"]=="train" and row["action_source"]=="expert"]
        fixed = batch(expert_rows[:min(4,len(expert_rows))],device)
    def stage_at(i):
        if smoke: return "joint"
        if variant=="policy_only": return "policy_only"
        if variant=="world": return "one" if i<steps*.25 else "world"
        return "one" if i<steps*.15 else "world" if i<steps*.45 else "policy" if i<steps*.6 else "joint"
    for i in range(start,steps):
        stage = stage_at(i)
        if smoke: x=fixed
        else:
            chosen=[]; seen=set()
            limit=4 if width>=768 else 32
            for index in rng.permutation(len(files)):
                pool=shard_rows(files[index])
                for j in rng.permutation(len(pool)):
                    row=pool[j]
                    if row["split"]!="train" or row["source_week"] in seen: continue
                    if stage in ("policy","policy_only") and row["action_source"]!="expert": continue
                    seen.add(row["source_week"]); chosen.append(decode(row))
                    if len(chosen)==limit: break
                if len(chosen)==limit: break
            if not chosen: raise ValueError("No eligible training rows")
            x=batch(chosen,device)
        model.train()
        rate = (1e-4 if stage=="joint" and not smoke else 3e-4) * min(1.,(i+1)/max(1.,steps*.05)) * (.1+.9*(1+np.cos(np.pi*i/max(steps,1)))/2)
        opt.param_groups[0]["lr"] = rate
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=="cuda"):
            loss,metrics=loss_for(model,x,stage)
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        done = i+1 == steps or time.monotonic() >= stop
        if i==start or (i+1)%10==0 or done:
            log=dict(step=i+1,stage=stage,loss=float(loss.detach()),**metrics)
            if val and not smoke:
                model.eval()
                with torch.no_grad():
                    log["validation"] = loss_for(model,batch(val[:16],device),"joint")[1]
            print(json.dumps(log),flush=True)
            history.append(log)
        if done or time.monotonic()-last_save>=300:
            tmp=latest.with_suffix(".tmp")
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),width=width,step=i+1,
                            rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if device=="cuda" else None,
                            history=history,variant=variant),tmp)
            tmp.replace(latest)
            (out/"training.json").write_text(json.dumps(history,indent=2))
            if commit: commit()
            last_save=time.monotonic()
        if done: break
    if smoke:
        if len(history)<2 or history[-1]["loss"]>=history[0]["loss"]: raise RuntimeError("Tiny batch loss did not improve")
    return str(latest)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("data");p.add_argument("output");p.add_argument("--width",type=int,default=64)
    p.add_argument("--steps",type=int,default=50);p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    train(a.data,a.output,a.width,a.steps,a.smoke)


if __name__=="__main__":
    main()
