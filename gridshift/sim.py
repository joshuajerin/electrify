"""One balanced AC simulator for rollout generation, evaluation and the UI."""
import copy
import math
from functools import lru_cache

import grid2op
import numpy as np
import pandapower as pp

try:
    import numba
except ImportError:
    numba = None

PRIMARY = "l2rpn_neurips_2020_track1_small"
TRANSFER = "l2rpn_case14_sandbox"
HOLD = {"kind": "hold"}
Q_RATIO = math.tan(math.acos(.95))


@lru_cache(maxsize=3)
def environment(name, smoke=False):
    return grid2op.make(name.removesuffix("_small") if smoke else name, test=smoke)


def scenarios(name, smoke=False):
    return [str(p).rsplit("/", 1)[-1] for p in environment(name, smoke).chronics_handler.subpaths]


@lru_cache(maxsize=6)
def source(name, episode, smoke):
    env = environment(name, smoke)
    env.set_id(episode)
    obs = env.reset()
    data = env.chronics_handler.real_data.data
    injections = tuple(np.asarray(getattr(data, field)).copy() for field in ("load_p", "load_q", "prod_p", "prod_v"))
    maintenance = np.zeros((len(injections[0]),env.n_line),dtype=bool)
    for field in ("maintenance","hazards"):
        flag = getattr(data,field,None)
        if flag is not None:
            maintenance |= np.asarray(flag,dtype=bool)
    profiles = (*injections,maintenance)
    meta = {key: np.asarray(getattr(env, key)).copy() for key in
            ("gen_pmin", "gen_pmax", "gen_max_ramp_up", "gen_max_ramp_down", "gen_redispatchable",
             "line_or_to_subid", "line_ex_to_subid", "gen_to_subid", "load_to_subid")}
    meta["ratings"] = env.get_thermal_limit().copy()
    reference = dict(v=obs.v_or.copy(), a=obs.a_or.copy(), p=obs.p_or.copy())
    return copy.deepcopy(env.backend._grid), profiles, meta, reference


class Simulator:
    def __init__(self, name=PRIMARY, episode=0, seed=0, smoke=False, start=0, request=5.):
        if name in ("singapore", "virginia"):
            from .regional_cases import simulation_source
            grid, self.profiles, self.meta, self.reference = simulation_source(name)
        else:
            grid, self.profiles, self.meta, self.reference = source(name, episode, smoke)
        self.net = copy.deepcopy(grid)
        self.name, self.episode, self.smoke = name, episode, smoke
        self.n = len(set(self.meta["gen_to_subid"]) | set(self.meta["load_to_subid"]) |
                     set(self.meta["line_or_to_subid"]) | set(self.meta["line_ex_to_subid"]))
        self.edges = np.stack((self.meta["line_or_to_subid"], self.meta["line_ex_to_subid"]))
        self.tick = min(start, len(self.profiles[0])-1)
        self.rng = np.random.default_rng(seed)
        self.request, self.cap = request, request
        self.bus = int(self.meta["load_to_subid"][0])
        self.base_loads = list(self.net.load.index)
        self.compute = pp.create_load(self.net, self.bus, p_mw=0.)
        self.offset = np.zeros(len(self.net.gen))
        self.cooldown = np.zeros(self.edges.shape[1], dtype=int)
        self.opened = np.zeros_like(self.cooldown, dtype=bool)
        self.rating_scale = np.ones_like(self.cooldown, dtype=float)
        self.demand_scale = 1.
        self.hidden_bus = -1
        self.hidden_scale = 1.
        self.mask_bus = -1
        self.event_schedule = {}
        self.history = []
        self.base_line = self.net.line.in_service.to_numpy().copy()
        self.base_trafo = self.net.trafo.in_service.to_numpy().copy()
        self.static_edges = np.concatenate([
            self.net.line[["r_ohm_per_km", "x_ohm_per_km", "length_km"]].to_numpy(),
            self.net.trafo[["vkr_percent", "vk_percent", "vn_hv_kv"]].to_numpy()
        ]).astype(np.float32)
        self.observe()

    def snapshot(self):
        return copy.deepcopy(self)

    def restore(self, snapshot):
        self.__dict__.update(copy.deepcopy(snapshot.__dict__))
        return self.observation

    def fork(self, snapshot=None, disturbance_seed=None):
        sim = copy.deepcopy(snapshot or self)
        if disturbance_seed is not None:
            sim.rng = np.random.default_rng(disturbance_seed)
        return sim

    def causal_fork(self, horizon=6):
        trial = self.fork()
        trial.event_schedule = {}
        profiles = [array.copy() for array in self.profiles]
        for array in profiles[:4]:
            array[self.tick+1:self.tick+1+horizon] = array[self.tick]
        trial.profiles = tuple(profiles)
        return trial

    def active(self, tick=None):
        i = min(self.tick if tick is None else tick, len(self.profiles[0])-1)
        return np.r_[self.base_line, self.base_trafo] & ~self.opened & ~self.profiles[4][i].astype(bool)

    def action_options(self, dispatch=True):
        opts = [HOLD] + [{"kind":"cap", "value":x} for x in (0., .25, .5, .75, 1.)]
        active = self.active()
        for i in range(len(active)):
            if self.cooldown[i] == 0 and not self.profiles[4][min(self.tick+1,len(self.profiles[0])-1),i]:
                opts.append({"kind":"line","a":i,"value":float(not active[i])})
        if dispatch:
            gens = np.flatnonzero(self.meta["gen_redispatchable"])
            opts.extend({"kind":"dispatch","a":int(a),"b":int(b),"value":s}
                        for a in gens for b in gens if a != b for s in (.01,.02,.05))
        return [a for a in opts if self.legal(a)]

    def legal(self, action):
        k = action["kind"]
        if k == "hold": return True
        if k == "cap": return 0 <= action["value"] <= 1
        if k == "line":
            i = action["a"]
            if not 0 <= i < len(self.opened) or self.cooldown[i] > 0: return False
            next_i = min(self.tick+1,len(self.profiles[0])-1)
            if self.profiles[4][next_i,i]: return False
            if action["value"]: return bool(self.opened[i])
            if not self.active()[i]: return False
            active = self.active().copy()
            active[i] = False
            connected = {int(self.net.gen.bus.iloc[np.flatnonzero(self.net.gen.slack.to_numpy())[0]])}
            for _ in range(self.n):
                expanded = connected | {int(v) for u,v in self.edges[:,active].T if u in connected} | {int(u) for u,v in self.edges[:,active].T if v in connected}
                if expanded == connected: break
                connected = expanded
            return all(b in connected for b in self.meta["load_to_subid"]) and self.bus in connected
        if k != "dispatch": return False
        a,b = action["a"], action["b"]
        if not (0 <= a < len(self.offset) and 0 <= b < len(self.offset) and a != b): return False
        if not all(self.meta["gen_redispatchable"][[a,b]]): return False
        i = min(self.tick+1,len(self.profiles[0])-1)
        baseline = self.profiles[2][i]
        delta = action["value"] * float(np.sum(baseline))
        return (0 < delta <= min(self.meta["gen_max_ramp_up"][a],self.meta["gen_max_ramp_down"][b])
                and baseline[a]+self.offset[a]+delta <= self.meta["gen_pmax"][a]
                and baseline[b]+self.offset[b]-delta >= self.meta["gen_pmin"][b])

    def greedy(self):
        peak = self.observation["loading"].max()
        fraction = self.cap / max(self.request, 1e-5)
        return {"kind":"cap","value":float(np.clip(fraction + (-.25 if peak >= .95 else .25),0,1))}

    def step(self, action):
        legal = self.legal(action)
        self.tick = min(self.tick+1, len(self.profiles[0])-1)
        self.cooldown = np.maximum(0, self.cooldown-1)
        for edit in self.event_schedule.get(self.tick, []):
            self.apply_edit(edit)
        if legal:
            k = action["kind"]
            if k == "cap": self.cap = self.request * action["value"]
            if k == "line":
                i = action["a"]
                if action["value"] and self.profiles[4][self.tick,i]: legal = False
                else:
                    self.opened[i] = not bool(action["value"])
                    self.cooldown[i] = 3
            if k == "dispatch":
                delta = action["value"] * self.profiles[2][self.tick].sum()
                self.offset[action["a"]] += delta
                self.offset[action["b"]] -= delta
        self.observe()
        self.observation["illegal"] = not legal
        return self.observation

    def apply_edit(self, edit):
        k,v = edit["kind"], edit["value"]
        i = edit.get("target",0)
        if k == "compute":
            full = self.cap >= self.request-1e-6
            self.request = max(0.,float(v))
            self.cap = self.request if full else min(self.cap,self.request)
        elif k == "bus": self.bus = int(v)
        elif k == "demand":
            if edit.get("hidden"):
                self.hidden_bus = int(i)
                self.mask_bus = int(i)
                self.hidden_scale = float(v)
            else: self.demand_scale = float(v)
        elif k == "rating": self.rating_scale[i] = float(v)
        elif k == "line": self.opened[i] = not bool(v)
        else: raise ValueError(f"Unknown edit {k}")

    def edit(self, intervention):
        self.apply_edit(intervention)
        return self.observe(replace=True)

    def observe(self, replace=False):
        net = self.net
        load_p,load_q,gen_p,gen_v = [a[self.tick].copy() for a in self.profiles[:4]]
        load_p *= self.demand_scale
        load_q *= self.demand_scale
        hidden = self.meta["load_to_subid"] == self.hidden_bus
        load_p[hidden] *= self.hidden_scale
        load_q[hidden] *= self.hidden_scale
        net.load.loc[self.base_loads, "p_mw"] = load_p
        net.load.loc[self.base_loads, "q_mvar"] = load_q
        net.load.loc[self.compute, ["bus","p_mw","q_mvar"]] = [self.bus, self.cap, self.cap*Q_RATIO]
        net.gen.loc[:, "p_mw"] = gen_p + self.offset
        net.gen.loc[:, "vm_pu"] = gen_v / net.bus.loc[net.gen.bus, "vn_kv"].to_numpy()
        active = self.active()
        nl = len(net.line)
        net.line.loc[:, "in_service"] = active[:nl]
        net.trafo.loc[:, "in_service"] = active[nl:]
        failure = False
        try:
            pp.runpp(net, numba=numba is not None, init="flat", check_connectivity=True, max_iteration=15)
        except (pp.LoadflowNotConverged, ValueError, np.linalg.LinAlgError):
            failure = True
        def branches(line_field,trafo_field):
            if failure: return np.zeros(len(active))
            return np.nan_to_num(np.r_[net.res_line[line_field].to_numpy(), net.res_trafo[trafo_field].to_numpy()])
        amps = 1000*branches("i_from_ka","i_hv_ka")
        loading = amps / np.maximum(self.meta["ratings"]*self.rating_scale, 1.)
        p = branches("p_from_mw","p_hv_mw")
        q = branches("q_from_mvar","q_hv_mvar")
        volts = np.zeros(self.n) if failure else np.nan_to_num(net.res_bus.vm_pu.to_numpy()[:self.n])
        served = 0. if failure or volts[self.bus] == 0 else self.cap
        load_bus = self.meta["load_to_subid"]
        gen_bus = self.meta["gen_to_subid"]
        demand = np.bincount(load_bus, weights=load_p,minlength=self.n)
        reactive = np.bincount(load_bus, weights=load_q,minlength=self.n)
        generation = np.bincount(gen_bus, weights=(gen_p+self.offset if failure else np.nan_to_num(net.res_gen.p_mw.to_numpy())),minlength=self.n)
        existing = float(np.maximum(load_p,0.)[volts[load_bus] > 0].sum())
        lost = float(np.maximum(load_p,0.).sum()) - existing
        gen_bad = False if failure else bool(np.any(net.res_gen.p_mw.to_numpy() > self.meta["gen_pmax"]+1e-3)
                                              or np.any(net.res_gen.p_mw.to_numpy() < self.meta["gen_pmin"]-1e-3))
        energized = volts > 0
        violation = bool(failure or lost > 1e-3 or gen_bad or np.any(loading>1.) or np.any((volts[energized]<.9)|(volts[energized]>1.1)))
        mask = np.ones(self.n)
        if self.mask_bus >= 0: mask[self.mask_bus] = 0
        cap = np.zeros(self.n); cap[self.bus] = self.cap/100
        req = np.zeros(self.n); req[self.bus] = self.request/100
        node = np.stack((demand*mask/100,reactive*mask/100,generation/100,volts,mask,cap,req,
                         np.bincount(gen_bus, weights=self.meta["gen_pmax"],minlength=self.n)/100,
                         np.bincount(gen_bus, weights=self.meta["gen_pmin"],minlength=self.n)/100),axis=1)
        edge = np.column_stack((np.log1p(np.abs(self.static_edges)),active,loading,p/100,q/100,
                                self.meta["ratings"]*self.rating_scale/1000,self.cooldown/3))
        angle = 2*np.pi*(self.tick%288)/288
        self.observation = dict(node=node.astype("float32"),edge=edge.astype("float32"),
            glob=np.array([np.sin(angle),np.cos(angle),self.request/100,self.cap/100],dtype="float32"),
            voltage=volts,loading=loading,flow=p,demand=demand,generation=generation,amps=amps,
            served=served,existing=existing,lost=lost,event=violation,failure=failure,active=active)
        if replace and self.history: self.history[-1] = self.observation
        else: self.history.append(self.observation)
        self.history = self.history[-12:]
        return self.observation
