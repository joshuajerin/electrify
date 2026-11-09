"""Wall-clock bounded training for allocation, control and resource outcomes."""
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .model_v2 import ResourceWAM

HISTORY = ("node","edge","glob","resource","previous","previous_maps")


def shard_index(path, limit=24):
    files = sorted(Path(path).glob("shard-*.npz"))
    if len(files)>limit:
        files = [files[i] for i in np.linspace(0,len(files)-1,limit,dtype=int)]
    loaded = []
    for f in files:
        if not f.with_suffix(".json").exists():
            continue
        with np.load(f,allow_pickle=False) as a:
            loaded.append(dict(a))
    return loaded


def sampler(shards,split):
    return [(shard,i) for shard in shards for i,k in enumerate(shard["split"]) if k==split]


def batch(rows,rng,device,count=2):
    selected, seen = [], set()
    for i in rng.choice(len(rows),min(len(rows),128),replace=False):
        shard,j = rows[i]
        episode = int(shard["episode"][j])
        if episode in seen:
            continue
        selected.append((shard,j));seen.add(episode)
        if len(selected)==count:
            break
    return {k:torch.from_numpy(np.stack([s[k][j] for s,j in selected])).to(device)
            for k in selected[0][0] if k not in ("split","seed","episode")}


def objective(model,x,horizon,policy=True,dynamics=True):
    links = x["links"][0].long()
    state = model.belief([x[k] for k in HISTORY],links)
    electrical, service, actions = [], [], []
    for t in range(horizon):
        if policy and t<3:
            logits = model.action_scores(state,x["option_maps"][:,t],x["options"][:,t])
            logits = logits.masked_fill(~x["option_mask"][:,t],-1e4)
            valid = x["labels"][:,t]>=0
            target = x["labels"][:,t].clamp_min(0).long()
            actions.append((F.cross_entropy(logits,target,reduction="none")*valid).sum()/valid.sum().clamp_min(1))
        state,(v,e,g,s) = model.imagine_step(state,x["actions"][:,t],x["maps"][:,t],links,x["contexts"][:,t])
        expected = x["global_target"][:,t]
        valid = 1-expected[:,4]
        voltage = (F.smooth_l1_loss(v,x["voltage"][:,t],reduction="none").mean(1)*valid).sum()/valid.sum().clamp_min(1)
        edge = (F.smooth_l1_loss(e,x["edge_target"][:,t],reduction="none").mean((1,2))*valid).sum()/valid.sum().clamp_min(1)
        n = g.shape[0]
        # Different heads omit different deterministic subsets of rows.
        mask = ((torch.arange(n,device=v.device)[:,None]+torch.arange(3,device=v.device)[None]+t)%4!=0).float()
        classification = F.binary_cross_entropy_with_logits(g[...,3:],expected[:,None,3:].expand_as(g[...,3:]),
                       pos_weight=g.new_tensor([2.,4.]),reduction="none")
        regression = F.smooth_l1_loss(g[...,:3],expected[:,None,:3].expand_as(g[...,:3]),reduction="none").mean(2)
        electrical.append(voltage+edge+2*(classification.mean(2)*mask).sum()/mask.sum().clamp_min(1)
                          +(regression*mask*valid[:,None]).sum()/(mask*valid[:,None]).sum().clamp_min(1))
        service.append(F.smooth_l1_loss(s,x["service"][:,t]/10.))
    world = torch.stack(electrical).mean()
    service_loss = torch.stack(service).mean()
    policy_loss = torch.stack(actions).mean() if actions else world.new_zeros(())
    if policy:
        allocation = model.belief([x[f"plan_{k}"] for k in HISTORY],links)
        acceptance,sites = model.heads(allocation,x["plan_resource"][:,-1,:,0].bool())
        policy_loss = policy_loss + F.cross_entropy(acceptance,x["capacity_target"].long())
        site_loss = -(sites.log_softmax(1)*x["site_target"]).sum(1)
        policy_loss = policy_loss + site_loss.mean()
    loss = (world+service_loss if dynamics else world.new_zeros(())) + policy_loss
    return loss,dict(world=float(world.detach()),service=float(service_loss.detach()),policy=float(policy_loss.detach()))


def train(path,out,warm,seconds,refresh=None,commit=None,failure_rows=None):
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(521)
    torch.manual_seed(521)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResourceWAM().to(device)
    params = sum(p.numel() for p in model.parameters())
    assert params >= 50_000_000
    payload = torch.load(warm,map_location="cpu",weights_only=True)
    model.load_state_dict(payload["model"],strict=payload.get("schema")==2)
    base,heads = [],[]
    for name,p in model.named_parameters():
        (heads if name.startswith(("resource_input","control_input","site_score","accept_score","service","candidate_score","context","policy")) else base).append(p)
    opt = torch.optim.AdamW([{"params":base,"lr":3e-5},{"params":heads,"lr":1e-4}],weight_decay=1e-4)
    loaded = shard_index(path)
    train_rows,val_rows = sampler(loaded,"train"),sampler(loaded,"val")
    assert train_rows and val_rows
    assert {int(s["episode"][j]) for s,j in train_rows}.isdisjoint(int(s["episode"][j]) for s,j in val_rows)
    val = batch(val_rows,rng,device)
    start = time.monotonic()
    step,last,logs = 0,0.,[]
    def save():
        tmp = out/"checkpoint.tmp"
        torch.save(dict(schema=2,model=model.state_dict(),width=768,step=step,parameters=params),tmp)
        tmp.replace(out/"checkpoint.pt")
        torch.save(dict(optimizer=opt.state_dict(),sampler=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device=="cuda" else [],step=step,
                        elapsed_s=time.monotonic()-start,stage=stage,normalization={"power_base_mw":100,"service_base":10},
                        shards=len(loaded)),out/"resume.pt")
        (out/"training.json").write_text(json.dumps(logs,indent=2))
        if commit: commit()
    while True:
        elapsed = time.monotonic()-start
        progress = elapsed/max(seconds,1)
        stage = "dynamics" if progress<.32 else "policy" if progress<.52 else "joint"
        if failure_rows:
            stage = "repair"
        for p in base: p.requires_grad_(stage!="policy")
        horizon = (1 if progress<.10 else 6) if stage=="dynamics" else 3 if stage=="policy" else 12
        opt.zero_grad(set_to_none=True)
        accumulate = 4 if device=="cuda" else 1
        for micro in range(accumulate):
            source = failure_rows if failure_rows and micro%2==0 else train_rows
            x = batch(source,rng,device)
            with torch.autocast(device,dtype=torch.bfloat16,enabled=device=="cuda"):
                loss,metrics = objective(model,x,horizon,stage!="dynamics",stage!="policy")
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite resource training loss")
            (loss/accumulate).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        step += 1
        elapsed = time.monotonic()-start
        if step==1 or step%15==0 or elapsed>=seconds:
            model.eval()
            with torch.no_grad(),torch.autocast(device,dtype=torch.bfloat16,enabled=device=="cuda"):
                validation = objective(model,val,6)[1]
            model.train()
            entry = dict(step=step,elapsed_s=round(elapsed),stage=stage,shards=len(loaded),
                         parameters=params,loss=float(loss.detach()),validation=validation,**metrics)
            logs.append(entry)
            print(json.dumps(entry),flush=True)
        if step==1 or elapsed-last>=110 or elapsed>=seconds:
            save()
            last=elapsed
            if refresh and elapsed<seconds:
                refresh()
                loaded=shard_index(path)
                train_rows=sampler(loaded,"train")
        if elapsed>=seconds:
            return str(out/"checkpoint.pt")
