"""Revision-bound connection experiments; every active branch outage is reported."""
import time
import numpy as np

from .judging import score


class Cancelled(Exception):
    pass


def study(sim, model, publish, cancelled):
    started = time.perf_counter()
    target = float(sim.request)
    outages = [int(i) for i in np.flatnonzero(sim.active())]
    solves = 0

    def check(bus, cap):
        nonlocal solves
        cases = []
        for outage in [None, *outages]:
            if cancelled():
                raise Cancelled()
            cases.append(score(sim, bus, cap, [] if outage is None else [outage], include_intact=outage is None)["cases"][-1])
            solves += 1
            if len(cases) % 10 == 0:
                publish(dict(progress=f"Bus {bus} · {cap:.1f} MW · case {len(cases)}/{len(outages)+1}", solves=solves))
        return cases

    publish(dict(stage="baseline", progress="Checking the existing grid before adding compute"))
    baseline = check(sim.bus, 0.)
    preexisting = [c["outage"] for c in baseline if not c["feasible"]]
    publish(dict(baseline=baseline, preexisting=preexisting, outages=outages, stage="propose"))
    nodes = [sim.bus, *[int(i) for i in np.argsort(-sim.observation["generation"]) if i != sim.bus][:2]]
    proposals = []

    def evaluate(bus, cap, origin, imagined=None):
        publish(dict(stage="verify", progress=f"Verifying {origin} · bus {bus} · {cap:.1f} MW"))
        cases = check(bus, cap)
        new = [c for base, c in zip(baseline, cases) if base["feasible"] and not c["feasible"]]
        unresolved = [c for c in cases if not c["feasible"]]
        worsened = [c for base, c in zip(baseline, cases) if not base["feasible"] and (
            c["lost"] > base["lost"] + .001 or c["loading"] > max(1., base["loading"]) + .001
            or (c["failure"] and not base["failure"]))]
        item = dict(id=len(proposals), bus=bus, cap_mw=cap, source=origin, cases=cases,
                    new_failures=len(new), unresolved=len(unresolved), worsened=len(worsened),
                    feasible=not unresolved, witness=(new or worsened or unresolved or [None])[0],
                    imagined=imagined)
        proposals.append(item)
        publish(dict(proposals=list(proposals), solves=solves))
        return item

    evaluate(sim.bus, target, "Original request")
    if model is not None:
        from .control import predict
        if cancelled(): raise Cancelled()
        publish(dict(stage="imagine", progress="WAM imagining five compute-cap alternatives over 30 minutes"))
        options, latency = predict(sim, model, caps_only=True)
        choice = options[0]
        evaluate(sim.bus, target * choice["action"]["value"], "WAM cap proposal",
                 dict(trajectory=choice["trajectory"], latency_ms=latency, source="learned",
                      scope="30-minute prediction; verification below is a current-point outage screen"))
    # Explicit engineering alternatives, kept separate from the learned proposal.
    for bus in nodes:
        if bus != sim.bus:
            evaluate(bus, target, "Relocate campus")
        for fraction in (.75, .5, .25):
            evaluate(bus, round(target * fraction, 3), "Reduced connection")
    verified = [p for p in proposals if p["feasible"]]
    relative = [p for p in proposals if p["new_failures"] == 0 and p["worsened"] == 0]
    best = max(verified or relative, key=lambda p: p["cap_mw"], default=None)
    publish(dict(stage="complete", progress="Connection comparison complete", best=best,
                 status="complete", solves=solves, duration_s=time.perf_counter()-started,
                 scope="One operating point, intact grid and every currently active branch outage. Three candidate buses; discrete caps. Baseline failures remain unresolved; this is not connection approval."))
