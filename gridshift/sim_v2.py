"""Synthetic capacity studies with electrically coupled workload and storage."""
from copy import deepcopy

import numpy as np
import pandapower as pp

from .sim import HOLD, Q_RATIO, Simulator

DT = 1/12
ETA = .95
ARRIVAL_STEPS = 24
TOTAL_STEPS = 36


def consume(queue, amount):
    """Process oldest work first. The first bucket expires after this step."""
    remaining = queue.copy()
    for i in range(len(queue)):
        take = min(remaining[i], amount)
        remaining[i] -= take
        amount -= take
    return remaining


def resource_plan(capacity, energy, queue, arrivals, available, processing, battery):
    """Return bounded requests, preserving immediate priority at each campus."""
    immediate = arrivals * .7
    room = np.maximum(0., np.clip(processing, 0., capacity) - immediate)
    available_work = queue.sum() + .3*arrivals.sum()*DT
    deferred = room * min(1., available_work/max(room.sum()*DT, 1e-12))
    pmax = capacity*.2*available
    maximum = capacity*.4
    charge = np.minimum(pmax, np.maximum(0., (.9*maximum-energy)/(ETA*DT)))
    discharge = np.minimum(pmax, np.maximum(0., (energy-.1*maximum)*ETA/DT))
    power = np.clip(battery, -charge, discharge)
    return immediate, deferred, power


class ResourceSimulator(Simulator):
    def __init__(self, *args, **kwargs):
        self.campuses = {}
        self.eligible = ()
        self.queue = np.zeros(12)
        self.duration = 0
        self.forced = set()
        self.disabled_batteries = set()
        self.causal = False
        super().__init__(*args, **kwargs)
        self.cap = 0.
        self.eligible = tuple(dict.fromkeys([self.bus, *map(int, self.meta["gen_to_subid"])]))[:3]
        if self.name in ("singapore", "virginia"):
            from .regional_cases import REGIONS
            region = REGIONS[self.name]
            self.eligible = region.sites
            self.model_power_scale = 650. / region.peak_mw
        self.observe(replace=True)

    def __deepcopy__(self, memo):
        clone = object.__new__(type(self))
        memo[id(self)] = clone
        shared = {"profiles", "meta", "reference", "edges", "static_edges", "base_line", "base_trafo"}
        clone.__dict__ = {k: v if k in shared else deepcopy(v, memo) for k, v in self.__dict__.items()}
        return clone

    def install(self, allocation, soc=.5):
        if self.duration or self.campuses:
            raise ValueError("Install on a fresh study fork")
        if not .1 <= soc <= .9 or len(allocation) > 3 or not set(allocation) <= set(self.eligible):
            raise ValueError("Invalid campus allocation or battery charge")
        if not all(np.isfinite(mw) and mw > 0 for mw in allocation.values()) or sum(allocation.values()) > self.request+1e-6:
            raise ValueError("Invalid capacity")
        self.cap = 0.
        for bus, mw in sorted(allocation.items()):
            self.campuses[bus] = dict(capacity_mw=mw, energy=mw*.4*soc, initial=mw*.4*soc,
                processing_mw=0., battery_mw=0., load=pp.create_load(self.net, bus, 0.),
                storage=pp.create_load(self.net, bus, 0.))
        self.observe(replace=True)

    def campus_view(self):
        total = sum(c["capacity_mw"] for c in self.campuses.values())
        return [dict(bus=b,capacity_mw=c["capacity_mw"],processing_mw=c["processing_mw"],
                     soc=c["energy"]/(.4*c["capacity_mw"]), battery_mw=c["battery_mw"],
                     energy_mwh=c["energy"],battery_enabled=b not in self.disabled_batteries,
                     backlog_mwh=float(self.queue.sum()*c["capacity_mw"]/total))
                for b,c in self.campuses.items()]

    def observe(self, replace=False):
        self.cap = 0.
        for c in self.campuses.values():
            self.net.load.loc[c["load"], ["p_mw","q_mvar"]] = [c["processing_mw"], c["processing_mw"]*Q_RATIO]
            self.net.load.at[c["storage"],"p_mw"] = -c["battery_mw"]
        o = super().observe(replace)
        o["campuses"] = self.campus_view()
        o["queue"] = self.queue.copy()
        o["gen_p"] = (self.net.res_gen.p_mw.to_numpy().copy() if not o["failure"]
                      else self.profiles[2][self.tick].copy())
        o["served"] = sum(c["processing_mw"] for b,c in self.campuses.items() if o["voltage"][b]>0)
        o["violations"] = self.violations(o)
        o["event"] = bool(o["violations"])
        return o

    def scenario_edit(self, kind, value, target=0, hidden=False):
        if kind == "eligible":
            if not value or len(value)>self.n or any(type(b) is not int or not 0<=b<self.n for b in value):
                raise ValueError("Invalid eligible buses")
            self.eligible = tuple(dict.fromkeys(value))
        elif kind == "battery":
            if not 0<=target<self.n:
                raise ValueError("Unknown battery bus")
            (self.disabled_batteries.discard if value else self.disabled_batteries.add)(target)
        elif kind == "line":
            if not 0<=target<len(self.opened):
                raise ValueError("Unknown branch")
            (self.forced.discard if value else self.forced.add)(target)
            self.opened[target] = not value
        elif kind == "compute":
            if self.campuses or not np.isfinite(value) or not 0<=value<=5000:
                raise ValueError("Invalid request or installed proposal")
            self.request = float(value)
        elif kind in ("rating", "demand"):
            if not np.isfinite(value) or not .1<=value<=2:
                raise ValueError("Invalid multiplier")
            if not 0<=target<(len(self.opened) if kind=="rating" else self.n):
                raise ValueError("Unknown target")
            self.apply_edit(dict(kind=kind,value=value,target=target,hidden=hidden))
        else:
            raise ValueError("Unknown edit")
        return self.observe(replace=True)

    def legal_grid(self, action):
        if action["kind"] == "line":
            if action["a"] in self.forced:
                return False
            if not action["value"]:
                if action["a"] in self.islanding(action["a"]):
                    return False
        return self.legal(action)

    def islanding(self, only=None):
        active = self.active()
        root = int(self.meta["gen_to_subid"][np.flatnonzero(self.net.gen.slack.to_numpy())[0]])
        def connected(skip):
            reachable = {root}
            edges = self.edges[:, active & (np.arange(len(active))!=skip)].T
            while True:
                new = reachable | {int(v) for u,v in edges if u in reachable} | {int(u) for u,v in edges if v in reachable}
                if new == reachable:
                    return reachable
                reachable = new
        intact = connected(-1)
        candidates = np.flatnonzero(active) if only is None else [only]
        result = {}
        for i in candidates:
            disconnected = intact-connected(i)
            if disconnected:
                result[int(i)] = sorted(disconnected)
        return result

    def default_action(self, mode="heuristic"):
        sites = self.campus_view()
        peak = float(self.observation["loading"].max())
        urgent = self.queue[:3].sum() > 1e-6
        return dict(processing=[c["capacity_mw"]*(1 if mode=="hold" or peak < .87 or urgent else .56) for c in sites],
                    battery=[c["capacity_mw"]*(.2 if peak>.96 else -.1 if peak<.8 and c["soc"]<.5 else 0.)
                             if mode!="hold" else 0. for c in sites], grid=HOLD)

    def violations(self, o):
        issues = []
        def add(kind, equipment, value, limit):
            issues.append(dict(kind=kind,equipment_id=equipment,timestep=int(self.tick),value=float(value),
                               limit=float(limit),severity=float(abs(value-limit))))
        if o["failure"]:
            add("solve_failure", None, 1, 0)
        else:
            for i in np.flatnonzero(o["loading"] > 1.0001):
                add("overload", int(i), o["loading"][i], 1)
            for i, v in enumerate(o["voltage"]):
                if 0 < v < .9 or v > 1.1:
                    add("voltage", i, v, .9 if v<.9 else 1.1)
            for i,p in enumerate(o["gen_p"]):
                lo,hi = self.meta["gen_pmin"][i],self.meta["gen_pmax"][i]
                if p<lo-.001 or p>hi+.001:
                    add("generation_limit",i,p,lo if p<lo else hi)
        for key in ("lost","immediate_lost_mwh","deadline_miss_mwh"):
            if o.get(key,0.) > .0001:
                add("existing_load_loss" if key=="lost" else key,None,o[key],0)
        if self.duration>=TOTAL_STEPS:
            for b,c in self.campuses.items():
                if c["energy"]<c["initial"]-1e-5:
                    add("terminal_battery",b,c["energy"],c["initial"])
        return issues

    def step(self, action):
        # Preview known availability changes before accepting the grid action.
        old_tick = self.tick
        next_tick = min(old_tick+1, len(self.profiles[0])-1)
        for event in self.event_schedule.get(next_tick, []):
            if event["kind"]=="line":
                branch = event["target"]
                (self.forced.discard if event["value"] else self.forced.add)(branch)
                self.opened[branch] = not event["value"]
            else:
                self.apply_edit(event)
        grid = action.get("grid",HOLD)
        accepted = self.legal_grid(grid)
        self.tick = next_tick
        self.cooldown = np.maximum(0, self.cooldown-1)
        if accepted and grid["kind"] == "line":
            self.opened[grid["a"]] = not grid["value"]
            self.cooldown[grid["a"]] = 3
        if accepted and grid["kind"] == "dispatch":
            change = grid["value"]*self.profiles[2][self.tick].sum()
            self.offset[grid["a"]] += change
            self.offset[grid["b"]] -= change
        cap = np.array([c["capacity_mw"] for c in self.campuses.values()])
        energy = np.array([c["energy"] for c in self.campuses.values()])
        arrivals = cap*.8 if self.duration<ARRIVAL_STEPS else cap*0
        p = np.asarray(action.get("processing",cap),dtype=float)
        b = np.asarray(action.get("battery",np.zeros_like(cap)),dtype=float)
        if p.shape!=cap.shape or b.shape!=cap.shape or not np.isfinite(p).all() or not np.isfinite(b).all():
            raise ValueError("Invalid resource action")
        immediate, deferred, power = resource_plan(cap,energy,self.queue,arrivals,
            np.array([i not in self.disabled_batteries for i in self.campuses]),p,b)
        immediate_served = np.minimum(immediate,np.clip(p,0,cap))
        for k,c in enumerate(self.campuses.values()):
            c["processing_mw"] = float(immediate_served[k]+deferred[k])
            c["battery_mw"] = float(power[k])
        self.duration += 1
        o = self.observe()
        online = np.array([not o["failure"] and o["voltage"][bus]>0 for bus in self.campuses],dtype=bool)
        for k,c in enumerate(self.campuses.values()):
            if online[k]:
                c["energy"] += float((-min(power[k],0)*ETA-max(power[k],0)/ETA)*DT)
            else:
                c["processing_mw"] = c["battery_mw"] = 0.
        queued = self.queue.copy()
        queued[-1] += .3*float(arrivals.sum())*DT
        left = consume(queued,float(deferred[online].sum())*DT)
        missed = float(left[0])
        self.queue = np.r_[left[1:],0.]
        executed = dict(processing=[c["processing_mw"] for c in self.campuses.values()],
                        battery=[c["battery_mw"] for c in self.campuses.values()],grid=grid if accepted else HOLD)
        completed = float((immediate_served[online]+deferred[online]).sum()*DT)
        o.update(served=completed/DT,completed_mwh=completed,
                 immediate_lost_mwh=float((immediate.sum()-immediate_served[online].sum())*DT),
                 deadline_miss_mwh=missed, proposed_action=action,executed_action=executed,
                 rejected=not accepted,clipped=not (np.allclose(executed["processing"],p) and np.allclose(executed["battery"],b)),
                 campuses=self.campus_view(),queue=self.queue.copy())
        o["violations"] = self.violations(o)
        o["event"] = bool(o["violations"])
        return o
