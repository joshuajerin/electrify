"""Capacity proposals, disturbance challenges and simulator evidence."""
import time

import numpy as np
import torch

from .model_v2 import control_features, history_arrays
from .sim import HOLD
from .sim_v2 import ARRIVAL_STEPS, DT, ETA, TOTAL_STEPS, consume, resource_plan


def sensitivity(sim):
    """DC injection-to-flow screening. AC still determines feasibility."""
    src,dst = sim.edges
    incidence = np.zeros((len(src),sim.n))
    incidence[np.arange(len(src)),src] = 1
    incidence[np.arange(len(src)),dst] = -1
    weight = np.where(sim.active(),1/np.maximum(.02,np.abs(sim.static_edges[:,1])),0)
    weighted = weight[:,None]*incidence
    return weighted @ np.linalg.pinv(incidence.T @ weighted)


def site_order(sim):
    sensitivity_matrix = sensitivity(sim)
    peak = sim.observation["loading"]
    slack = int(sim.meta["gen_to_subid"][np.flatnonzero(sim.net.gen.slack.to_numpy())[0]])
    return sorted(sim.eligible, key=lambda i: (np.sum(np.abs(sensitivity_matrix[:,slack]-sensitivity_matrix[:,i])*
                                                   np.maximum(peak-.6,0)),i))


def candidates(sim):
    sites = sim.campus_view()
    cap = [s["capacity_mw"] for s in sites]
    hold, heuristic = sim.default_action("hold"),sim.default_action()
    choices = [hold,heuristic]
    for fraction, power in ((.56,.2),(1.,.2),(1.,-.1),(.8,0.)):
        choices.append(dict(processing=[c*fraction for c in cap],
                            battery=[c*power for c in cap],grid=HOLD))
    for bus in range(len(cap)):
        limits = cap.copy()
        limits[bus] *= .56
        choices.append(dict(processing=limits,battery=heuristic["battery"],grid=HOLD))
    gens = np.flatnonzero(sim.meta["gen_redispatchable"])
    grid = [dict(kind="dispatch",a=int(i),b=int(j),value=value)
            for i in gens for j in gens if i!=j for value in (.01,.02,.05)]
    grid = [a for a in grid if sim.legal(a)]
    if grid:
        dc = sensitivity(sim)
        line = int(sim.observation["loading"].argmax())
        flow = float(sim.observation["flow"][line])
        gen = sim.meta["gen_to_subid"]
        grid.sort(key=lambda a: (dc[line,gen[a["a"]]]-dc[line,gen[a["b"]]])*a["value"]*np.sign(flow))
        choices.extend(dict(processing=cap,battery=heuristic["battery"],grid=a) for a in grid[:2])
    near = np.argsort(-sim.observation["loading"])[:3]
    for i in near:
        a = dict(kind="line",a=int(i),value=0.)
        if sim.legal_grid(a):
            choices.append(dict(processing=cap,battery=heuristic["battery"],grid=a))
            break
    return choices


def ordered(o):
    return (int(o["failure"]),o["lost"],o["immediate_lost_mwh"],o["deadline_miss_mwh"],
            len(o["violations"]), sum(v["severity"] for v in o["violations"]),-o["completed_mwh"])


def flex_step(sim):
    """Probe the declared next study interval and defer only flexible work."""
    best = None
    for solves,fraction in enumerate((1., .68, .56),1):
        trial = sim.fork()
        action = trial.default_action("hold")
        action["processing"] = [site["capacity_mw"]*fraction for site in trial.campus_view()]
        observation = trial.step(action)
        rank = ordered(observation)
        if best is None or rank < best[0]:
            best = rank,trial,observation
        if not observation["violations"]:
            break
    return best[1],best[2],solves


def teacher(sim, options, width=4):
    trial = sim.fork()
    trial.event_schedule = {}
    # The teacher sees a persistence forecast, not the hidden disturbance schedule.
    profiles = list(sim.profiles)
    for k,p in enumerate(profiles):
        copy = p.copy()
        copy[sim.tick+1:sim.tick+2] = p[sim.tick]
        profiles[k] = copy
    trial.profiles = tuple(profiles)
    ids = list(dict.fromkeys([0,1,*range(max(2,len(options)-width+2),len(options))]))
    return min(ids,key=lambda i:ordered(trial.fork().step(options[i])))


def model_arrays(sim,model):
    device = next(model.parameters()).device
    arrays = [torch.from_numpy(x).unsqueeze(0).to(device) for x in history_arrays(sim)]
    return arrays,torch.as_tensor(sim.edges,device=device)


def option_arrays(sim, options):
    a,m = zip(*(control_features(sim,o) for o in options))
    return np.stack(a),np.stack(m)


def reserve_battery_command(requested, capacity, energy, available, reserve, future_steps):
    """Limit this step's dispatch to what the remaining charge window can repay."""
    maximum_charge_mw = .2 * capacity * available
    floor = reserve - future_steps * maximum_charge_mw * ETA * DT
    margin = energy - floor
    # Positive power discharges. A negative margin requires charging now.
    upper_mw = np.where(margin >= 0, margin * ETA / DT, margin / (ETA * DT))
    return np.minimum(requested, np.maximum(upper_mw, -maximum_charge_mw))


@torch.inference_mode()
def model_policy(sim, model, policy_only=False, horizon=12):
    arrays,links = model_arrays(sim,model)
    state = model.belief(arrays,links)
    options = candidates(sim)
    a,m = option_arrays(sim,options)
    device = links.device
    aa,mm = torch.from_numpy(a).to(device),torch.from_numpy(m).to(device)
    logits = model.action_scores(state,mm[None],aa[None])[0]
    order = logits.argsort(descending=True).tolist()
    if policy_only:
        return options[order[0]], []
    indexes = list(dict.fromkeys([0,1,*order]))[:8]
    count = len(indexes)
    expanded = tuple(t.expand(count,*t.shape[1:]) for t in state)
    frames = [[] for _ in indexes]
    scores = torch.zeros(count,device=device)
    scale = getattr(sim, "model_power_scale", 1.)
    sites = sim.campus_view()
    capacity = np.array([c["capacity_mw"] for c in sites])
    available = np.array([c["battery_enabled"] for c in sites])
    reserve = np.array([sim.campuses[c["bus"]]["initial"] for c in sites])
    energy = np.tile([c["energy_mwh"] for c in sites],(count,1))
    queue = np.tile(sim.queue,(count,1))
    resource_choices = [i for i,o in enumerate(options) if o["grid"]["kind"]=="hold"]
    chosen = indexes.copy()
    first_actions = []
    for step in range(horizon):
        if step in (1,2):
            ranked = model.action_scores(expanded,mm[None].expand(count,-1,-1,-1),aa[None].expand(count,-1,-1))
            chosen = [resource_choices[i] for i in ranked[:,resource_choices].argmax(1).tolist()]
        moves, maps = [], []
        for j,i in enumerate(chosen):
            cmd = options[i]
            arrivals = capacity*.8 if sim.duration+step<ARRIVAL_STEPS else capacity*0
            future_steps = max(0, TOTAL_STEPS-sim.duration-step-1)
            battery = reserve_battery_command(cmd["battery"], capacity, energy[j], available, reserve, future_steps)
            immediate,deferred,power = resource_plan(capacity,energy[j],queue[j],arrivals,available,
                                                       cmd["processing"],battery)
            if step == 0:
                first_actions.append(dict(cmd, battery=power.tolist()))
            immediate_delivered = np.minimum(immediate,np.clip(cmd["processing"],0,capacity))
            energy[j] += (-np.minimum(power,0)*ETA-np.maximum(power,0)/ETA)*DT
            queue[j,-1] += .3*arrivals.sum()*DT
            left = consume(queue[j],deferred.sum()*DT)
            scores[j] += float((immediate-immediate_delivered).sum()*DT+left[0])*5*scale
            queue[j] = np.r_[left[1:],0.]
            action = dict(processing=immediate_delivered+deferred,battery=power,
                          grid=cmd["grid"] if step==0 else HOLD)
            move,local = control_features(sim,action)
            moves.append(move);maps.append(local)
            future_sites = [dict(c,processing_mw=float(action["processing"][k]),battery_mw=float(power[k]),
                                 energy_mwh=float(energy[j,k]),soc=float(energy[j,k]/(.4*capacity[k])),
                                 backlog_mwh=float(queue[j].sum()*capacity[k]/capacity.sum()))
                            for k,c in enumerate(sites)]
            frames[j].append(dict(tick=sim.tick+step+1,campuses=future_sites,violations=[]))
        condition = torch.stack((expanded[2].squeeze(-1),
                                  torch.as_tensor(sim.rating_scale,device=device,dtype=torch.float32)[None].expand(count,-1)),dim=-1)
        expanded,(v,e,g,s) = model.imagine_step(expanded,torch.as_tensor(np.stack(moves),device=device),
                                               torch.as_tensor(np.stack(maps),device=device),links,condition)
        scores += (e[...,0]-1).clamp_min(0).sum(1)+(.9-v).clamp_min(0).sum(1)
        scores += g[...,3:].mean(1).sigmoid().sum(1) + s[:,1:3].clamp_min(0).sum(1)*5 - s[:,0]*.1
        volts,loads = v.tolist(),e[...,0].tolist()
        for j in range(count):
            frames[j][-1].update(loading=loads[j],voltage=volts[j])
    if sites:
        terminal = np.maximum(0.,reserve[None]-energy).sum(1)
        scores += torch.as_tensor(terminal,device=device,dtype=scores.dtype)*scale*(.4 if sim.duration+horizon<TOTAL_STEPS else 5)
    best = int(scores.argmin())
    return first_actions[best],frames[best]


@torch.inference_mode()
def learned_allocation(sim,model):
    arrays,links = model_arrays(sim,model)
    state = model.belief(arrays,links)
    accepted,scores = model.heads(state,arrays[3][:,-1,:,0].bool())
    units = int(accepted.argmax())
    buses = scores[0].argsort(descending=True).tolist()[:min(3,len(sim.eligible),units)]
    if not buses:
        return {}
    weights = torch.softmax(scores[0,buses],0).cpu().numpy()
    raw = weights*units
    counts = raw.astype(int)
    for i in np.argsort(-(raw-counts))[:units-counts.sum()]:
        counts[i] += 1
    return {bus:float(sim.request*count/20) for bus,count in zip(buses,counts) if count}


def frame(sim,o):
    return dict(tick=int(sim.tick),loading=o["loading"].tolist(),voltage=o["voltage"].tolist(),
                campuses=o["campuses"],violations=o.get("violations",[]))


def replay(sim, allocation, case=None, policy="heuristic", model=None, cancelled=lambda:False, steps=TOTAL_STEPS):
    trial = sim.fork()
    trial.install(allocation,soc=.5 if case is None else case.get("soc",.5))
    if case and "edits" in case:
        for delay,event in case["edits"]:
            trial.event_schedule.setdefault(trial.tick+delay,[]).append(event)
    frames, imagined = [], []
    totals = dict(completed_mwh=0.,immediate_lost_mwh=0.,deadline_miss_mwh=0.,existing_lost_mwh=0.,violations=0)
    solves = 0
    for i in range(steps):
        if cancelled():
            raise InterruptedError()
        if policy=="flex":
            trial,o,count = flex_step(trial)
            solves += count
        elif policy in ("learned","policy_only") and model is not None:
            action, forecast = model_policy(trial,model,policy=="policy_only")
            if i==0:
                imagined = forecast
        elif policy == "ac":
            options = candidates(trial)
            action = options[teacher(trial,options)]
            solves += min(4,len(options))
        else:
            action = trial.default_action(policy)
        if policy!="flex":
            o = trial.step(action)
            solves += 1
        frames.append(frame(trial,o))
        for key in ("completed_mwh","immediate_lost_mwh","deadline_miss_mwh"):
            totals[key] += o[key]
        totals["existing_lost_mwh"] += o["lost"]*DT
        totals["violations"] += len(o["violations"])
    totals["battery_terminal_mwh"] = sum(c["energy"] for c in trial.campuses.values())
    violations = [v for f in frames for v in f["violations"]]
    return dict(trajectory=frames,imagined=imagined,metrics=totals,witness=next(iter(violations),None),
                passed=not violations,solves=solves)


def catalogue(sim):
    islands = sim.islanding()
    intact = [{"id":"intact"}]
    outages = [{"id":f"outage-{i}","edits":[(4,dict(kind="line",target=int(i),value=False))]}
               for i in np.flatnonzero(sim.active()) if i not in islands]
    return intact+outages,islands


def study(sim,model,publish,cancelled,budget=1600,cases=None):
    started = time.perf_counter()
    all_cases,islands = catalogue(sim)
    # Rank outages by flow at the decision state and preregister a bounded suite.
    scope = "Declared study scenarios; islanding separately reported."
    if cases is None:
        cases = all_cases[:1]+sorted(all_cases[1:],key=lambda c:-sim.observation["loading"][c["edits"][0][1]["target"]])[:4]
        critical = int(sim.observation["loading"].argmax())
        cases.append(dict(id="stress",soc=.3,edits=[(6,dict(kind="rating",target=critical,value=.85))]))
        scope = "Bounded AC screen: intact, four ranked connected outages and a derating stress; islanding separately reported."
    if budget<len(cases)*TOTAL_STEPS:
        raise ValueError("AC solve budget cannot cover the baseline cases")
    solves = 0
    baseline = {}
    proposals, activity = [], []
    def event(kind,message,proposal=None):
        activity.append(dict(kind=kind,message=message,proposal=proposal))
        publish(dict(progress=message,activity=list(activity),solves=solves))
    event("baseline","Checking existing service before adding compute")
    for case in cases:
        result = replay(sim,{},case,cancelled=cancelled)
        solves += result["solves"]
        baseline[case["id"]] = result["witness"]
    eligible = [case for case in cases if baseline[case["id"]] is None]
    publish(dict(baseline=baseline,islanding=islands,supported_outages=len(all_cases)-1,
                 tested_cases=[c["id"] for c in cases],scope=scope))
    if not eligible:
        publish(dict(status="complete",best=None,progress="All tested baseline cases fail",solves=solves))
        return
    buses = site_order(sim)
    target = sim.request
    specs = [({buses[0]:target},"Immediate processing reference","hold")]
    if model is not None:
        allocation = learned_allocation(sim,model)
        specs.append((allocation,"WAM allocation","learned"))
        if not allocation and hasattr(sim, "model_power_scale"):
            # Preserve the model's zero-capacity result, but let users also
            # compare its dispatch policy at an independently selected site.
            specs.append(({buses[0]:target},"WAM dispatch on first site","learned"))
    if len(buses)>1:
        specs.append(({buses[0]:target*.55,buses[1]:target*.45},"Sensitivity-ranked split","heuristic"))
    incumbent = None
    attempted = set()
    repair_stage = 0
    pending_flex = None
    while len(proposals)<6:
        parent = None
        if pending_flex is not None:
            parent,pending_flex = pending_flex,None
            allocation = {p["bus"]:p["mw"] for p in parent["allocation"]}
            source,policy = "Flexible workload repair","flex"
        elif specs:
            allocation,source,policy=specs.pop(0)
        else:
            if incumbent and incumbent["accepted_mw"]>=target-1e-6:
                break
            failed = [p for p in proposals if p["witness"] and p["accepted_mw"]>0]
            if not failed or repair_stage>=3:
                break
            parent = min(failed,key=lambda p:(-p["accepted_mw"],p["grade"]["failed_cases"],
                                              p["grade"]["violations"],-p["grade"]["completed_mwh"]))
            total = parent["accepted_mw"]
            if repair_stage==0:
                allocation = {p["bus"]:p["mw"] for p in parent["allocation"]}
                source,policy = "Flexible workload repair","flex"
            else:
                feedback = parent["witness"]
                ranked = list(buses)
                if feedback["kind"]=="overload":
                    failed_case = next(c for c in eligible if c["id"]==parent["case"])
                    grid = sim.fork()
                    for delay,edit in failed_case.get("edits",[]):
                        if edit["kind"]=="line" and sim.tick+delay<=feedback["timestep"]:
                            grid.opened[edit["target"]] = not edit["value"]
                    dc = sensitivity(grid)
                    line = feedback["equipment_id"]
                    ranked.sort(key=lambda b:abs(dc[line,b]))
                if repair_stage==2:
                    total = .8*total
                allocation = {ranked[0]:total} if len(ranked)<2 else {ranked[0]:total*.7,ranked[1]:total*.3}
                source,policy = ("Witness-guided relocation" if repair_stage==1 else "Witness-guided reduction"),"heuristic"
            repair_stage += 1
        allocation = {b:mw for b,mw in allocation.items() if mw>0}
        worst_solves = len(eligible)*TOTAL_STEPS*(3 if policy=="flex" else 1)
        if solves+worst_solves>budget:
            if specs or (parent and repair_stage<3):
                continue
            break
        key = (policy,tuple(sorted((b,round(mw,6)) for b,mw in allocation.items())))
        if key in attempted:
            continue
        attempted.add(key)
        index = len(proposals)
        event("propose",f"{source}: {sum(allocation.values()):.1f} MW",index)
        results = []
        for case in eligible:
            result = replay(sim,allocation,case,policy,model,cancelled)
            solves += result["solves"]
            results.append((case,result))
        # A successful repair should replay its parent's failing condition so
        # the before-and-after traces use the same scenario.
        representative = next(((c,r) for c,r in results if parent and c["id"]==parent["case"]),results[0])
        case,result = next(((c,r) for c,r in results if not r["passed"]),representative)
        passed = all(r["passed"] for _,r in results)
        witness = result["witness"]
        if witness:
            event("challenge",f"{case['id']} revealed {witness['kind']}",index)
        grade = dict(failed_cases=sum(not r["passed"] for _,r in results),
                     violations=sum(r["metrics"]["violations"] for _,r in results),
                     completed_mwh=sum(r["metrics"]["completed_mwh"] for _,r in results),
                     deadline_miss_mwh=sum(r["metrics"]["deadline_miss_mwh"] for _,r in results))
        item = dict(id=index,allocation=[dict(bus=b,mw=mw) for b,mw in allocation.items()],
                    accepted_mw=sum(allocation.values()),source=source,verified=True,passed=passed,
                    policy=policy,case=case["id"],scenario=case,witness=witness,grade=grade,
                    parent_id=parent["id"] if parent else None,
                    conditional=any(value is not None for value in baseline.values()),trajectory=result["trajectory"],
                    imagined=result["imagined"],metrics=result["metrics"],
                    cases=[dict(id=c["id"],passed=r["passed"],witness=r["witness"],metrics=r["metrics"]) for c,r in results])
        proposals.append(item)
        if policy=="hold" and witness and witness["kind"]=="overload":
            pending_flex = item
        event("grade",f"{grade['failed_cases']}/{len(results)} cases failed · {grade['violations']} violations",index)
        if parent:
            event("update",f"Proposal {parent['id']+1} → {index+1}: {parent['grade']['failed_cases']} → {grade['failed_cases']} failed cases",index)
        if passed and (incumbent is None or item["accepted_mw"]>incumbent["accepted_mw"]):
            incumbent=item
        publish(dict(proposals=list(proposals),best=incumbent,solves=solves))
    publish(dict(status="complete",progress="Study complete",solves=solves,duration_s=time.perf_counter()-started))
