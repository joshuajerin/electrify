"""Independent, steady-state regional study preview."""
from pathlib import Path
import math

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandapower as pp
from pydantic import BaseModel, Field

from gridshift.regional_cases import REGIONS, make_network
from gridshift.api_v2 import router


app = FastAPI(title="Grid Shift regional preview")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5174", "http://localhost:5174"],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
app.include_router(router)


class StudyRequest(BaseModel):
    requested_mw: float = Field(gt=0, le=5000)
    demand_scale: float = Field(default=1, ge=.75, le=1.15)


@app.get("/")
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


def screen(net, branches, load_ids, region, allocation):
    for ix, mw in zip(load_ids, allocation):
        net.load.loc[ix, ["p_mw", "q_mvar"]] = [mw, mw*math.tan(math.acos(.95))]
    contingency_type, contingency_ix = branches[region.outage]
    worst = None
    for outage in (False, True):
        net[contingency_type].at[contingency_ix, "in_service"] = not outage
        condition = "Corridor outage" if outage else "Intact grid"
        try:
            pp.runpp(net, numba=False, init="flat", calculate_voltage_angles=True)
        except pp.LoadflowNotConverged:
            return False, {"condition": condition, "equipment": "AC power flow", "value": None}
        for i, voltage in net.res_bus.vm_pu.items():
            if not math.isfinite(voltage) or not .90 <= voltage <= 1.10:
                return False, {"condition": condition, "equipment": region.buses[i], "value": None if not math.isfinite(voltage) else round(voltage, 3), "unit": "p.u."}
        for i, (kind, idx) in enumerate(branches):
            if outage and i == region.outage:
                continue
            loading = float(net[f"res_{kind}"].at[idx, "loading_percent"])
            if not math.isfinite(loading):
                return False, {"condition": condition, "equipment": "Branch loading", "value": None}
            if worst is None or loading > worst["raw"]:
                a, b, _, _ = region.corridors[i]
                worst = {"condition": condition, "equipment": f"{region.buses[a]} → {region.buses[b]}",
                         "value": round(loading, 1), "unit": "% loaded", "raw": loading}
    return worst.pop("raw") <= 100, worst


@app.get("/api/regions")
def list_regions():
    return [{"id": key, "name": r.name, "peak_mw": r.peak_mw, "reference": r.reference, "source": r.source}
            for key, r in REGIONS.items()]


@app.post("/api/regions/{key}/study")
def study(key: str, request: StudyRequest):
    region = REGIONS.get(key)
    if region is None:
        raise HTTPException(404, "Unknown region")
    net, branches, loads = make_network(region, request.demand_scale)
    baseline_ok, baseline = screen(net, branches, loads, region, (0, 0))
    options = []
    for name, shares in ((region.buses[region.sites[0]], (1, 0)), (region.buses[region.sites[1]], (0, 1)), ("Split evenly", (.5, .5))):
        full_ok, full = screen(net, branches, loads, region, [s*request.requested_mw for s in shares])
        accepted = request.requested_mw if full_ok and baseline_ok else 0
        passing_case = full if accepted else baseline
        if baseline_ok and not full_ok:
            for step in range(19, 0, -1):
                trial = request.requested_mw * step / 20
                ok, witness = screen(net, branches, loads, region, [s*trial for s in shares])
                if ok:
                    accepted, passing_case = trial, witness
                    break
        options.append({"name": name, "accepted_mw": round(accepted, 2), "requested_case": full, "passing_case": passing_case,
                        "sites": [{"name": region.buses[bus], "mw": round(accepted*share, 2)}
                                  for bus, share in zip(region.sites, shares) if share]})
    best = max(options, key=lambda item: (item["accepted_mw"],
        -(item["passing_case"]["value"] if item["accepted_mw"] else 0)))
    a, b, _, _ = region.corridors[region.outage]
    return {"region": region.name, "requested_mw": request.requested_mw, "baseline_ok": baseline_ok,
            "baseline": baseline, "options": options, "best": best,
            "demand_mw": round(region.peak_mw*request.demand_scale), "voltage_kv": [region.high_kv, 230],
            "outage": f"{region.buses[a]} → {region.buses[b]}", "bus_count": len(region.buses),
            "corridor_count": len(region.corridors)}
