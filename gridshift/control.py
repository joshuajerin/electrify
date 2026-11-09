import time
from pathlib import Path

import numpy as np
import torch

from .model import WAM, action_features, tensor_history
from .sim import HOLD


def load_model(path, device="cpu"):
    if not path or not Path(path).is_file(): return None
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    model=WAM(ckpt["width"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def public_frame(o):
    return {k:o[k].tolist() if isinstance(o[k],np.ndarray) else o[k]
            for k in ("loading","voltage","served","existing","lost","event","failure")}


def candidate_options(sim):
    opts=sim.action_options()
    return opts


@torch.inference_mode()
def predict(sim, model, memory=True, policy_only=False, world_only=False, caps_only=False):
    start=time.perf_counter()
    if model is None:
        candidate=[]
        for a in (HOLD,sim.greedy()):
            trial=sim.fork()
            trajectory=[public_frame(trial.step(a if t==0 else HOLD)) for t in range(6)]
            candidate.append(dict(action=a,trajectory=trajectory,score=0.,source="simulator"))
        return candidate,(time.perf_counter()-start)*1000
    device=next(model.parameters()).device
    links=torch.as_tensor(sim.edges,device=device)
    history=sim.history if memory else sim.history[-1:]
    nodes,edges,glob=tensor_history(history,device)
    if not memory: nodes,edges,glob=nodes[:,-1:],edges[:,-1:],glob[:,-1:]
    state=model.encode(nodes,edges,glob,links)
    opts=[a for a in candidate_options(sim) if a["kind"]=="cap"] if caps_only else candidate_options(sim)
    feat=torch.as_tensor([action_features(sim,a) for a in opts],device=device,dtype=torch.float32)
    if caps_only:
        plans=[[a,HOLD,HOLD] for a in opts]
    elif world_only:
        candidates = opts[:6] + [a for a in opts[6:] if a["kind"]=="dispatch"][:2]
        plans = [[a,HOLD,HOLD] for a in candidates]
    else:
        scores=model.logits(state,feat[None])[0]
        best=torch.topk(scores,min(8,len(opts))).indices.tolist()
        plans=[[opts[i]] for i in best]
    for depth in (() if world_only or caps_only else (1,2)):
        prefixes=torch.as_tensor([[action_features(sim,a) for a in plan] for plan in plans],device=device,dtype=torch.float32)
        expanded=tuple(x.expand(len(plans),*x.shape[1:]) for x in state)
        chosen=model.logits(expanded,feat[None].expand(len(plans),-1,-1),prefixes).argmax(1).tolist()
        for plan,index in zip(plans,chosen): plan.append(opts[index])
    if not caps_only:
        plans += [[HOLD]*3,[sim.greedy(),HOLD,HOLD]]
    encoded=torch.as_tensor([[action_features(sim,a) for a in plan+[HOLD]*3] for plan in plans],device=device,dtype=torch.float32)
    stacked=tuple(x.expand(len(plans),*x.shape[1:]) for x in state)
    v,e,g=model.rollout(stacked,encoded,links)
    pred=g.mean(2)
    prob=pred[...,3:].sigmoid().cpu().numpy()
    pred=pred.cpu().numpy(); v=v.cpu().numpy(); e=e.cpu().numpy()
    uncertainty=g.std(2).mean((1,2)).cpu().numpy()
    result=[]
    for i,plan in enumerate(plans):
        traj=[]
        for j in range(6):
            loading=e[i,j,:,0].clip(0)
            voltage=v[i,j]
            served=float(np.clip(pred[i,j,0]*100,0,sim.request))
            lost=float(max(0.,pred[i,j,2]*100))
            traj.append(dict(loading=loading.tolist(),voltage=voltage.tolist(),served=served,
                             existing=float(max(0.,pred[i,j,1]*100)),lost=lost,event=bool(prob[i,j,0]>.5),
                             failure=bool(prob[i,j,1]>.5),event_probability=float(prob[i,j,0])))
        switches=sum(a["kind"]=="line" for a in plan)
        rank=(sum(x["failure"] for x in traj),round(sum(x["lost"] for x in traj),1),
              sum(x["event"] for x in traj),-sum(x["served"] for x in traj),switches)
        result.append(dict(action=plan[0],plan=plan,trajectory=traj,rank=rank,
                           score=float(sum(x["served"] for x in traj)/12),disagreement=float(uncertainty[i]),source="learned"))
    if not policy_only: result.sort(key=lambda item:item["rank"])
    return result,(time.perf_counter()-start)*1000


def assisted(sim,candidates):
    for i,proposal in enumerate(candidates):
        action=proposal["action"]
        if not sim.legal(action): continue
        obs=sim.fork().step(action)
        if not obs["event"]: return action,i
    return HOLD,len(candidates)
