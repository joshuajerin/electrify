"""Generate independently seeded counterfactual windows and Parquet shards."""
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .model import action_features, frame
from .sim import HOLD, PRIMARY, Simulator, scenarios


def ordered(obs, action):
    return (int(obs["failure"]),round(obs["lost"],4),int(obs["event"]),-obs["served"],int(action["kind"]=="line"))


def expert(sim):
    options = sim.action_options()
    caps = [a for a in options if a["kind"] in ("hold","cap")]
    others = [a for a in options if a["kind"] not in ("hold","cap")]
    sample = others[:3]
    if len(others)>3:
        sample = [others[i] for i in sim.rng.choice(len(others),3,replace=False)]
    candidates = caps+sample
    forecast = sim.causal_fork(1)
    best = min(range(len(candidates)),key=lambda i: ordered(forecast.fork().step(candidates[i]),candidates[i]))
    return candidates[best],candidates,best


def split_map(name=PRIMARY, smoke=False):
    names = scenarios(name,smoke)
    ids = np.random.default_rng(401).permutation(len(names))
    a,b = max(1,int(.7*len(ids))),max(2,int(.85*len(ids)))
    return {key: list(map(int,x)) for key,x in zip(("train","val","test"),(ids[:a],ids[a:b],ids[b:]))}


def initial(name, episode, seed, smoke, hidden_history=False):
    rng = np.random.default_rng(seed)
    sim = Simulator(name,episode,seed,smoke,request=0.)
    stop = len(sim.profiles[0])
    sim.tick = int(rng.integers(0,max(1,stop-32)))
    sim.edit({"kind":"bus","value":int(rng.choice(sim.meta["load_to_subid"]))})
    baseline = np.maximum(sim.profiles[0][sim.tick],0).sum()
    sim.edit({"kind":"compute","value":float(baseline*rng.uniform(0,.10))})
    sim.edit({"kind":"demand","value":float(rng.uniform(.8,1.2))})
    sim.history.clear()
    sim.observe()
    if hidden_history:
        disturbance(sim,2,"train",rng,int(rng.integers(1,10)))
    for _ in range(11): sim.step(HOLD)
    return sim


def disturbance(sim, kind, split, rng, delay=1):
    if kind == 0: return
    edge_count = len(sim.opened)
    if kind == 1:
        ids = np.arange(edge_count)
        ids = ids[ids%5 != 0] if split == "train" else ids[ids%5 == 0]
        event = {"kind":"line","target":int(rng.choice(ids)),"value":False}
    elif kind == 2:
        event = {"kind":"demand","target":int(rng.choice(sim.meta["load_to_subid"])),
                 "value":float(rng.choice([.8,.9,1.1,1.2])),"hidden":True}
    else:
        event = {"kind":"rating","target":int(rng.integers(edge_count)),"value":float(rng.uniform(.7,1.))}
    sim.event_schedule.setdefault(sim.tick+delay,[]).append(event)


def window(sim, behavior):
    hist = sim.history
    obs = {key:np.stack([o[key] for o in hist]) for key in ("node","edge","glob")}
    obs["links"] = sim.edges
    actions,targets,options,labels = [],[],[],[]
    for t in range(6):
        choices = [HOLD]
        index = -1
        if behavior == "expert" and t < 3:
            action,choices,index = expert(sim)
        elif behavior == "greedy": action = sim.greedy()
        elif behavior == "random":
            avail = sim.action_options()
            action = avail[int(sim.rng.integers(len(avail)))]
        else: action = HOLD
        actions.append(action_features(sim,action))
        encoded = [action_features(sim,a) for a in choices]
        options.append(encoded+[[0,0,0,0]]*(9-len(encoded)))
        labels.append(index)
        targets.append(sim.step(action))
    obs.update(actions=np.array(actions,np.float32),options=np.array(options,np.float32),
               labels=np.array(labels),voltage=np.stack([o["voltage"] for o in targets]).astype(np.float32),
               edge_target=np.stack([o["edge"][:,4:7] for o in targets]),
               global_target=np.stack([frame(o) for o in targets]))
    buf = io.BytesIO()
    np.savez_compressed(buf,**obs)
    return buf.getvalue()


def generate(path, shard, count, smoke=False):
    path = Path(path); path.mkdir(parents=True,exist_ok=True)
    output = path/f"shard-{shard:03d}.parquet"
    if output.exists(): return str(output)
    splits = split_map(smoke=smoke)
    rng = np.random.default_rng(7351+shard)
    rows = []
    for i in range(count):
        split = ("test" if i%10==0 else "val" if i%10==1 else "train")
        if not splits[split]: split = "train"
        episode = int(rng.choice(splits[split]))
        seed = int(rng.integers(2**31))
        mode=(i+shard)%8
        sim = initial(PRIMARY,episode,seed,smoke,hidden_history=mode==4)
        if mode in (5,6,7):
            if mode == 5:
                pool=[a for a in sim.action_options(dispatch=False) if a["kind"]=="line" and not a["value"]
                      and (a["a"]%5!=0 if split=="train" else a["a"]%5==0)]
                if pool:
                    sim.edit({"kind":"line","target":pool[int(rng.integers(len(pool)))]["a"],"value":False})
            elif mode == 6:
                candidates=np.argsort(-sim.observation["loading"])[:8]
                sim.edit({"kind":"rating","target":int(rng.choice(candidates)),"value":float(rng.uniform(.7,.95))})
            else:
                sim.edit({"kind":"bus","value":int(rng.choice(sim.meta["load_to_subid"]))})
            # Include post-edit readings in the model's history before it proposes.
            for _ in range(int(rng.integers(1,5))): sim.step(HOLD)
        else:
            disturbance(sim,mode if mode<4 else 0,split,rng)
        for behavior in ("hold","greedy","expert","random"):
            rows.append(dict(split=split,source_week=episode,seed=seed,branch_parent_id=f"{shard}:{i}",
                             action_source=behavior,payload=window(sim.fork(),behavior)))
    tmp = output.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows),tmp)
    tmp.replace(output)
    return str(output)


def manifest(path,smoke):
    path=Path(path)
    sources=scenarios(PRIMARY,smoke)
    payload=dict(network=PRIMARY,smoke=smoke,files=[p.name for p in sorted(path.glob("shard-*.parquet"))],
                 splits={k:[sources[i] for i in ids] for k,ids in split_map(smoke=smoke).items()})
    (path/"manifest.json").write_text(json.dumps(payload,indent=2))
    return payload
