"""Public demand references and illustrative regional AC cases."""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import pandapower as pp


@dataclass(frozen=True)
class Region:
    name: str
    peak_mw: int
    reference: str
    source: str
    buses: tuple
    high_kv: int
    loads: tuple
    generation: tuple
    # Endpoints, corridor rating in MVA, series reactance on a 1,000 MVA base.
    corridors: tuple
    outage: int
    sites: tuple


REGIONS = {
    "singapore": Region(
        "Singapore", 8600, "EMA system peak through August 2026",
        "https://www.ema.gov.sg/news-events/news/media-releases/2026/five-year-infrastructure-plan-to-meet-unpredecented-future-demand",
        ("West supply", "South supply", "West exchange", "Central exchange", "West campus", "North campus", "East demand"),
        400, (0, 0, .08, .10, .19, .23, .40), ((1, .38), (2, .10)),
        ((0, 1, 5500, .021), (0, 2, 6600, .018), (1, 3, 5700, .019), (2, 3, 5000, .024),
         (2, 4, 4200, .015), (3, 5, 5700, .013), (4, 5, 2750, .025), (5, 6, 6000, .016),
         (4, 6, 2950, .026)),
        6, (4, 5),
    ),
    "virginia": Region(
        "Northern Virginia", 24271, "2025 Dominion zone summer peak reference",
        "https://www.pjm.com/-/media/DotCom/documents/ferc/filings/2026/20260507-el26-63-000.pdf",
        ("Regional supply", "West corridor", "South corridor", "North hub", "Ashburn campus", "West campus", "Other Dominion demand", "East hub"),
        500, (0, .02, .03, .06, .17, .09, .48, .15), ((1, .24), (2, .26), (7, .14)),
        ((0, 1, 10800, .010), (0, 2, 10800, .013), (1, 3, 8900, .010), (2, 7, 10500, .012),
         (3, 7, 7900, .014), (3, 4, 6400, .011), (1, 5, 6900, .013), (4, 5, 3100, .020),
         (2, 6, 15000, .009), (7, 6, 13000, .012)),
        4, (4, 5),
    ),
}

def make_network(region, scale):
    net = pp.create_empty_network(sn_mva=1000)
    for i, name in enumerate(region.buses):
        kv = 230 if i in region.sites else region.high_kv
        pp.create_bus(net, vn_kv=kv, name=name)
    pp.create_ext_grid(net, 0, vm_pu=1.025)
    for bus, share in region.generation:
        pp.create_gen(net, bus, p_mw=share * region.peak_mw * scale, vm_pu=1.02)
    for bus, share in enumerate(region.loads):
        if share:
            mw = region.peak_mw * share * scale
            pp.create_load(net, bus, p_mw=mw, q_mvar=mw * math.tan(math.acos(.98)))
    branches = []
    for a, b, rating, x in region.corridors:
        voltage = net.bus.vn_kv.at[a]
        if voltage == net.bus.vn_kv.at[b]:
            base_z = voltage**2 / net.sn_mva
            ix = pp.create_line_from_parameters(
                net, a, b, length_km=1, r_ohm_per_km=.08*x*base_z,
                x_ohm_per_km=x*base_z, c_nf_per_km=0,
                max_i_ka=rating/(math.sqrt(3)*voltage),
            )
            branches.append(("line", ix))
        else:
            hi, lo = sorted((a, b), key=lambda i: net.bus.vn_kv.at[i], reverse=True)
            vk = 100*x*rating/net.sn_mva
            ix = pp.create_transformer_from_parameters(
                net, hi, lo, sn_mva=rating, vn_hv_kv=net.bus.vn_kv.at[hi],
                vn_lv_kv=net.bus.vn_kv.at[lo], vk_percent=vk, vkr_percent=.08*vk,
                pfe_kw=0, i0_percent=0,
            )
            branches.append(("trafo", ix))
    load_ids = [pp.create_load(net, bus, p_mw=0) for bus in region.sites]
    return net, branches, load_ids


# Coordinates are schematic and use the same scene units as the benchmark.
LAYOUTS = {
    "singapore": ((-9, 2), (-5, 5), (-6, -1), (-1, 2), (-3, -3), (2, -5), (8, -1)),
    "virginia": ((-9, 0), (-5, -4), (-4, 5), (0, -4), (5, -5), (-1, 0), (5, 6), (5, 1)),
}
DEFAULT_REQUEST = {"singapore": 1000., "virginia": 4000.}


@lru_cache(maxsize=2)
def simulation_source(name):
    region = REGIONS[name]
    net, branches, compute_loads = make_network(region, 1.)
    net.load.drop(index=compute_loads, inplace=True)
    # The resource simulator accounts for the balancing source as a generator.
    net.ext_grid.drop(net.ext_grid.index, inplace=True)
    scheduled = region.peak_mw * (1 - sum(share for _, share in region.generation))
    pp.create_gen(net, 0, p_mw=scheduled, vm_pu=1.025, slack=True)
    rated = {(kind, ix): mva for (kind, ix), (_, _, mva, _) in zip(branches, region.corridors)}
    ratings = [rated["line", i] * 1000 / (math.sqrt(3) * net.bus.vn_kv.at[row.from_bus])
               for i, row in net.line.iterrows()]
    ratings += [rated["trafo", i] * 1000 / (math.sqrt(3) * row.vn_hv_kv)
                for i, row in net.trafo.iterrows()]
    production = net.gen.p_mw.to_numpy()
    meta = dict(
        gen_pmin=np.zeros(len(production)), gen_pmax=production + region.peak_mw*.18,
        gen_max_ramp_up=np.full(len(production), region.peak_mw*.02),
        gen_max_ramp_down=np.full(len(production), region.peak_mw*.02),
        gen_redispatchable=~net.gen.slack.to_numpy(),
        line_or_to_subid=np.r_[net.line.from_bus, net.trafo.hv_bus].astype(int),
        line_ex_to_subid=np.r_[net.line.to_bus, net.trafo.lv_bus].astype(int),
        gen_to_subid=net.gen.bus.to_numpy().astype(int),
        load_to_subid=net.load.bus.to_numpy().astype(int),
        ratings=np.asarray(ratings),
    )
    rows = 288  # One day of stationary peak conditions, with five-minute replay steps.
    profiles = tuple(np.tile(values, (rows, 1)) for values in (
        net.load.p_mw.to_numpy(), net.load.q_mvar.to_numpy(), production,
        net.gen.vm_pu.to_numpy() * net.bus.vn_kv.loc[net.gen.bus].to_numpy(),
        np.zeros(len(ratings), dtype=bool),
    ))
    return net, profiles, meta, {}
