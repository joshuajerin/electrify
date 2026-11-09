"""Graph recurrent dynamics and an autoregressive equipment-aware policy."""
import numpy as np
import torch
from torch import nn


def action_features(sim, action):
    kinds = {"hold":0,"cap":1,"dispatch":2,"line":3}
    k = kinds[action["kind"]]
    a,b = action.get("a",0),action.get("b",0)
    if k == 2: a,b = map(int,sim.meta["gen_to_subid"][[a,b]])
    elif k == 1: a = sim.bus
    return [k,a,b,action.get("value",0.)]


def frame(obs):
    return np.array([obs["served"]/100, obs["existing"]/100, obs["lost"]/100,
                     obs["event"],obs["failure"]],dtype=np.float32)


def mlp(a,b,d):
    return nn.Sequential(nn.Linear(a,d),nn.SiLU(),nn.Linear(d,b))


class GraphBlock(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.msg = mlp(3*d,d,d)
        self.update = mlp(2*d,d,d)
        self.node_norm = nn.LayerNorm(d)
        self.edge_norm = nn.LayerNorm(d)

    def forward(self,n,e,links,active):
        src,dst = links
        fwd = self.msg(torch.cat((n[:,src],n[:,dst],e),dim=-1))
        back = self.msg(torch.cat((n[:,dst],n[:,src],e),dim=-1))
        active = active.to(n.dtype)
        pooled = torch.zeros_like(n).index_add(1,dst,(fwd*active).to(n.dtype)).index_add(1,src,(back*active).to(n.dtype))
        counts = torch.zeros_like(n[:,:,:1]).index_add(1,dst,active).index_add(1,src,active)
        return self.node_norm(n+self.update(torch.cat((n,pooled/counts.clamp_min(1)),dim=-1))),self.edge_norm(e+(fwd+back)*.5)


class WAM(nn.Module):
    def __init__(self,d=128):
        super().__init__()
        self.d = d
        self.node_in = mlp(9,d,d)
        self.edge_in = mlp(9,d,d)
        self.glob_in = mlp(4,d,d)
        self.action_in = mlp(5,d,d)
        self.blocks = nn.ModuleList(GraphBlock(d) for _ in range(5 if d >= 768 else 3))
        self.node_mem = nn.GRUCell(d,d)
        self.edge_mem = nn.GRUCell(d,d)
        self.transition = GraphBlock(d)
        self.node_out = mlp(d,1,d)
        self.edge_out = mlp(d,3,d)
        self.global_out = nn.ModuleList(mlp(2*d,5,d) for _ in range(3))
        self.policy_in = mlp(2*d,d,d)
        layer = nn.TransformerEncoderLayer(d,4,2*d,dropout=0.,batch_first=True,norm_first=True)
        self.policy = nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.policy_score = mlp(2*d,1,d)

    def embed_actions(self,a):
        return self.action_in(torch.cat((nn.functional.one_hot(a[...,0].long(),4).float(),a[...,3:]),dim=-1))

    def action_maps(self,a,n,e):
        emb = self.embed_actions(a)
        kind,i,j = a[...,:3].long().unbind(-1)
        one = lambda ix,count: nn.functional.one_hot(ix.clamp(0,count-1),count).unsqueeze(-1)
        an = (one(i,n.shape[1])*(kind[:,None,None]==1)
              +(one(i,n.shape[1])-one(j,n.shape[1]))*(kind[:,None,None]==2))*emb[:,None,:]
        ae = one(i,e.shape[1])*(kind[:,None,None]==3)*emb[:,None,:]
        return an,ae,emb

    def encode(self, node,edge,glob,links,previous=None,node_extra=None,edge_extra=None):
        b,t,n,_ = node.shape
        e = edge.shape[2]
        hn = node.new_zeros((b,n,self.d))
        he = edge.new_zeros((b,e,self.d))
        for i in range(t):
            gn = self.glob_in(glob[:,i])[:,None]
            x,y = self.node_in(node[:,i])+gn,self.edge_in(edge[:,i])+gn
            if node_extra is not None: x = x + node_extra[:,i]
            if edge_extra is not None: y = y + edge_extra[:,i]
            if previous is not None:
                an,ae,_ = self.action_maps(previous[:,i],x,y)
                x,y = x+an,y+ae
            for block in self.blocks:
                x,y = block(x,y,links,edge[:,i,:,3:4])
            hn = self.node_mem(x.flatten(0,1),hn.flatten(0,1)).reshape(b,n,-1)
            he = self.edge_mem(y.flatten(0,1),he.flatten(0,1)).reshape(b,e,-1)
        return hn,he,edge[:,-1,:,3:4]

    def advance(self,state,a,links,node_extra=None,edge_extra=None):
        n,e,active = state
        an,ae,emb = self.action_maps(a,n,e)
        selected = nn.functional.one_hot(a[:,1].long().clamp(0,e.shape[1]-1),e.shape[1]).unsqueeze(-1).bool()
        active = torch.where(selected & (a[:,0,None,None]==3), a[:,3,None,None].expand_as(active),active)
        xn,xe = self.transition(n+an+emb[:,None]+(0 if node_extra is None else node_extra),
                                e+ae+(0 if edge_extra is None else edge_extra),links,active)
        n = self.node_mem(xn.flatten(0,1),n.flatten(0,1)).reshape_as(n)
        e = self.edge_mem(xe.flatten(0,1),e.flatten(0,1)).reshape_as(e)
        g = torch.cat((n.mean(1),e.mean(1)),dim=-1)
        heads = torch.stack([head(g) for head in self.global_out],dim=1)
        return (n,e,active),(self.node_out(n).squeeze(-1),self.edge_out(e)*active,heads)

    def rollout(self,state,actions,links):
        values = []
        for a in actions.unbind(1):
            state,out = self.advance(state,a,links)
            values.append(out)
        return tuple(torch.stack(x,dim=1) for x in zip(*values))

    def logits(self,state,options,prefix=None):
        n,e,_ = state
        global_token = self.policy_in(torch.cat((n.mean(1),e.mean(1)),dim=-1))[:,None]
        tokens = global_token if prefix is None else torch.cat((global_token,self.embed_actions(prefix)),dim=1)
        mask = torch.ones((tokens.shape[1],tokens.shape[1]),device=tokens.device,dtype=torch.bool).triu(1)
        q = self.policy(tokens,mask=mask)[:,-1]
        k,i,j = options[...,:3].long().unbind(-1)
        read = lambda x,idx: x.gather(1,idx.clamp(0,x.shape[1]-1)[...,None].expand(-1,-1,self.d))
        location = read(n,i)*(k[...,None]==1) + (read(n,i)-read(n,j))*(k[...,None]==2) + read(e,i)*(k[...,None]==3)
        options = self.embed_actions(options) + location
        return self.policy_score(torch.cat((options,q[:,None].expand_as(options)),dim=-1)).squeeze(-1)


def tensor_history(history,device):
    history = [history[0]]*(12-len(history))+history[-12:]
    return [torch.from_numpy(np.stack([o[k] for o in history])[None]).to(device) for k in ("node","edge","glob")]
