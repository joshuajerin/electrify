"""Capacity and operating-policy heads over the electrical world model."""
import numpy as np
import torch
from torch import nn

from .model import WAM, action_features, mlp
from .sim import HOLD

RESOURCE_DIM = 21


def resource_features(sim, obs):
    scale = getattr(sim, "model_power_scale", 1.) / 100
    x = np.zeros((sim.n,RESOURCE_DIM),dtype=np.float32)
    x[list(sim.eligible),0] = 1
    x[:,1] = sim.request*scale
    x[:,2] = obs["node"][:,4]
    for c in obs["campuses"]:
        b = c["bus"]
        x[b,3:9] = [c["capacity_mw"]*scale,c["processing_mw"]*scale,c["soc"],
                     c["battery_enabled"], c["battery_mw"]*scale, c["energy_mwh"]*scale]
        x[b,9:] = obs["queue"]*scale
    return x


def control_features(sim, action):
    scale = getattr(sim, "model_power_scale", 1.) / 100
    x = np.zeros((sim.n,4),dtype=np.float32)
    for k,bus in enumerate(sim.campuses):
        x[bus,:2] = [action["processing"][k]*scale,action["battery"][k]*scale]
    grid = action.get("grid",HOLD)
    if grid["kind"]=="dispatch":
        a,b = (int(sim.meta["gen_to_subid"][grid[k]]) for k in ("a","b"))
        x[a,2] += grid["value"]
        x[b,2] -= grid["value"]
    x[:,3] = 1
    return np.asarray(action_features(sim,grid),dtype=np.float32),x


def history_arrays(sim):
    history = [sim.history[0]]*max(0,12-len(sim.history))+sim.history[-12:]
    old = [np.stack([o[k] for o in history]) for k in ("node","edge","glob")]
    scale = getattr(sim, "model_power_scale", 1.)
    if scale != 1.:
        # Preserve topology, voltages and percent loading. Rebase only quantities
        # represented in MW, Mvar or amps in the checkpoint's inputs.
        old[0][..., [0, 1, 2, 5, 6, 7, 8]] *= scale
        old[1][..., [5, 6, 7]] *= scale
        old[2][..., 2:4] *= scale
    resources = np.stack([resource_features(sim,o) for o in history])
    previous, action_maps = zip(*(control_features(sim,o["executed_action"])
        if "executed_action" in o and len(o["executed_action"]["processing"])==len(sim.campuses)
        else (np.zeros(4,np.float32),np.zeros((sim.n,4),np.float32)) for o in history))
    return (*old,resources,np.stack(previous),np.stack(action_maps))


class ResourceWAM(WAM):
    def __init__(self,width=768):
        super().__init__(width)
        self.resource_input = mlp(RESOURCE_DIM,width,width)
        self.control_input = mlp(4,width,width)
        self.site_score = mlp(width,1,width)
        self.accept_score = mlp(2*width,21,width)
        self.service = mlp(2*width,5,width)
        self.candidate_score = mlp(3*width,1,width)
        self.context = mlp(2,width,width)

    def belief(self, arrays, links):
        node, edge, glob, resource, previous, maps = arrays
        extras = self.resource_input(resource) + self.control_input(maps)
        return self.encode(node,edge,glob,links,previous,node_extra=extras)

    def heads(self,state,eligible):
        n,e,_ = state
        pooled = torch.cat((n.mean(1),e.mean(1)),dim=-1)
        return self.accept_score(pooled), self.site_score(n).squeeze(-1).masked_fill(~eligible,-1e4)

    def imagine_step(self,state,action,maps,links,scenario=None):
        node_condition = self.control_input(maps)
        edge_condition = None if scenario is None else self.context(scenario)
        next_state, (v,e,g) = self.advance(state,action,links,node_condition,edge_condition)
        n,h,_ = next_state
        service = self.service(torch.cat((n.mean(1),h.mean(1)),dim=-1))
        return next_state,(v,e,g,service)

    def action_scores(self,state,maps,actions):
        n,e,_ = state
        enc = self.control_input(maps)
        score = torch.cat((n.mean(1)[:,None].expand(-1,enc.shape[1],-1),
                           e.mean(1)[:,None].expand(-1,enc.shape[1],-1),
                           (enc*n[:,None]).mean(2)),dim=-1)
        return self.candidate_score(score).squeeze(-1) + self.logits(state,actions)


def load_resource_checkpoint(path, device="cpu"):
    payload = torch.load(path,map_location="cpu",weights_only=True)
    if payload.get("schema")!=2:
        raise ValueError("A v2 resource checkpoint is required")
    m = ResourceWAM(payload["width"])
    m.load_state_dict(payload["model"])
    return m.to(device).eval()
