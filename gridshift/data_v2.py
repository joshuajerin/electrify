"""Immutable resource rollout shards. Each row belongs to a source episode."""
import json
from pathlib import Path

import numpy as np

from .data import split_map
from .model import frame
from .model_v2 import control_features, history_arrays
from .planner_v2 import candidates, option_arrays, site_order, teacher
from .sim import PRIMARY
from .sim_v2 import ResourceSimulator

K = 12


def make_window(episode, seed, split, smoke=False, actor=None):
    rng = np.random.default_rng(seed)
    sim = ResourceSimulator(PRIMARY,episode,seed,smoke,request=0.)
    sim.tick = int(rng.integers(0,max(1,len(sim.profiles[0])-64)))
    sim.observe(replace=True)
    sim.request = float(sim.observation["demand"].clip(0).sum()*rng.uniform(.02,.09))
    sim.demand_scale = float(rng.uniform(.88,1.10))
    order = site_order(sim)
    x = int(rng.integers(8,21))
    n = min(len(order),int(rng.integers(1,4)))
    counts = rng.multinomial(x,[1/n]*n)
    allocation = {b:float(sim.request*c/20) for b,c in zip(order,counts) if c}
    sim.install(allocation,soc=float(rng.uniform(.25,.75)))
    sim.history.clear()
    sim.observe()
    for _ in range(11):
        sim.step(sim.default_action())
    i = int(rng.integers(len(sim.opened)))
    bus = int(rng.choice(sim.meta["load_to_subid"]))
    mode = int(seed%5)
    events = [
        dict(kind="rating",target=i,value=.82),
        dict(kind="line",target=i,value=False),
        dict(kind="demand",target=bus,value=1.16,hidden=True),
    ]
    if mode<3:
        sim.event_schedule[sim.tick+4] = [events[mode]]
    if mode==3 and allocation:
        sim.disabled_batteries.add(next(iter(allocation)))
        sim.observe(replace=True)
    arrays = history_arrays(sim)
    # A causal, AC-scored allocation label on an independent zero-compute fork.
    initial = ResourceSimulator(PRIMARY,episode,seed,smoke,start=sim.tick,request=sim.request)
    initial.demand_scale = sim.demand_scale
    initial.observe(replace=True)
    ranked = site_order(initial)
    scored = []
    for units,weights in ((20,[20]),(20,[11,9]),(16,[9,7]),(12,[12]),(8,[8]),(0,[])):
        if len(weights)>len(ranked):
            continue
        trial = initial.fork()
        trial.install({b:trial.request*w/20 for b,w in zip(ranked,weights)})
        o = trial.step(trial.default_action())
        scored.append((bool(o["violations"]),-units,weights))
    bad,negative,weights = min(scored,key=lambda a:(a[0],a[1]))
    sites = np.zeros(sim.n,np.float32)
    if not bad and weights:
        for b,w in zip(ranked,weights):
            sites[b] = w/sum(weights)
    plan_arrays = history_arrays(initial)
    actions,maps,contexts,volt,edges,glob,service,opts,optmaps,masks,labels = ([] for _ in range(11))
    behavior = int(seed%4)
    for step in range(12):
        choices = candidates(sim)
        aa,mm = option_arrays(sim,choices)
        label = teacher(sim,choices) if behavior>=2 or actor else -1
        selected = (0 if behavior==0 else int(rng.integers(len(choices))) if behavior==1 else label)
        contexts.append(np.column_stack((sim.active(),sim.rating_scale)).astype(np.float32))
        o = sim.step(actor(sim) if actor else choices[selected])
        a,m = control_features(sim,o["executed_action"])
        actions.append(a);maps.append(m)
        volt.append(o["voltage"]);edges.append(o["edge"][:,4:7]);glob.append(frame(o))
        service.append([o["completed_mwh"],o["immediate_lost_mwh"],o["deadline_miss_mwh"],
                        o["queue"].sum(),sum(c["energy_mwh"] for c in o["campuses"])])
        opts.append(np.pad(aa,((0,K-len(aa)),(0,0))))
        optmaps.append(np.pad(mm,((0,K-len(mm)),(0,0),(0,0))))
        masks.append(np.arange(K)<len(aa))
        labels.append(label if not o["rejected"] else -1)
    row = {k:v for k,v in zip(("node","edge","glob","resource","previous","previous_maps"),arrays)}
    row.update({f"plan_{k}":v for k,v in zip(("node","edge","glob","resource","previous","previous_maps"),plan_arrays)})
    row.update(actions=actions,maps=maps,contexts=contexts,voltage=volt,edge_target=edges,global_target=glob,
               service=service,options=opts,option_maps=optmaps,option_mask=masks,labels=labels,
               links=sim.edges,site_target=sites,capacity_target=(-negative if not bad else 0),
               episode=episode,seed=seed,split=split)
    return {k:np.asarray(v,dtype=np.float32 if k not in ("split","episode","seed","links","labels","capacity_target","option_mask") else None)
            for k,v in row.items()}


def generate_shard(directory, index, count=32, smoke=False):
    path = Path(directory)
    path.mkdir(parents=True,exist_ok=True)
    dest = path/f"shard-{index:04d}.npz"
    if dest.exists():
        return str(dest)
    rng = np.random.default_rng(92511+index)
    splits = split_map(smoke=smoke)
    rows = []
    for i in range(count):
        split = "val" if i%8==0 and splits["val"] else "train"
        source = int(rng.choice(splits[split]))
        rows.append(make_window(source,int(rng.integers(2**31)),split,smoke))
    tmp = dest.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
    tmp.replace(dest)
    metadata = {"schema":2,"file":dest.name,"windows":count,"transitions":count*12,
                "source_episodes":sorted({int(row["episode"]) for row in rows})}
    dest.with_suffix(".json").write_text(json.dumps(metadata,indent=2))
    return str(dest)
