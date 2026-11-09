"""Verify native benchmark import and counterfactual determinism."""
import argparse
import json
import grid2op
import numpy as np

from .sim import HOLD, PRIMARY, Simulator


def check(smoke=True):
    # Compare the AC solver on the same topology. Native random attacks and
    # automatic overload trips would otherwise change only the reference grid.
    env=grid2op.make(PRIMARY.removesuffix("_small") if smoke else PRIMARY, test=smoke,
                    opponent_init_budget=0., opponent_budget_per_ts=0.)
    env.parameters.NO_OVERFLOW_DISCONNECTION = True
    max_amps=max_flow=max_voltage=0.
    checked=0
    count=len(env.chronics_handler.subpaths)
    for episode in range(min(count,8)):
        sim=Simulator(PRIMARY,episode,0,smoke,request=0)
        env.set_id(episode)
        obs=env.reset()
        for _ in range(100):
            o=sim.observation
            max_amps=max(max_amps,float(np.max(np.abs(o["amps"]-obs.a_or))))
            max_flow=max(max_flow,float(np.max(np.abs(o["flow"]-obs.p_or))))
            expected_v=obs.v_or/env.backend._grid.bus.loc[env.line_or_to_subid,"vn_kv"].to_numpy()
            max_voltage=max(max_voltage,float(np.max(np.abs(o["voltage"][env.line_or_to_subid]-expected_v)[obs.line_status],initial=0)))
            checked+=1
            if checked>=100: break
            obs,_,done,_=env.step(env.action_space())
            if done: break
            sim.step(HOLD)
        if checked>=100: break
    assert checked==100,(checked,max_amps,max_flow,max_voltage)
    assert max_amps<.1 and max_flow<.01 and max_voltage<1e-4,(max_amps,max_flow,max_voltage)
    trial=Simulator(PRIMARY,0,0,smoke)
    a,b=trial.fork(),trial.fork()
    for _ in range(3):
        np.testing.assert_array_equal(a.step(HOLD)["loading"],b.step(HOLD)["loading"])
    return dict(checked=checked,max_error_amps=max_amps,max_error_mw=max_flow,max_error_voltage_pu=max_voltage)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--full",action="store_true")
    print(json.dumps(check(not p.parse_args().full)))
