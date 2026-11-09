"""Run with python -m tests.check_resource_sim or directly with PYTHONPATH=."""
import numpy as np

from gridshift import check
from gridshift.planner_v2 import reserve_battery_command
from gridshift.sim_v2 import DT, ETA, ResourceSimulator

check.Simulator = ResourceSimulator
print("native parity", check.check())
s = ResourceSimulator(smoke=True,request=20)
s.install({s.eligible[0]:10.,s.eligible[1]:10.})
fork = s.fork()
o = fork.step(dict(processing=[5.6,5.6],battery=[2,-2]))
assert np.isclose(o["completed_mwh"],11.2*DT)
assert np.isclose(o["queue"].sum(),4.8*DT)
assert np.isclose(fork.campus_view()[0]["energy_mwh"],2-2*DT/ETA)
assert np.isclose(fork.campus_view()[1]["energy_mwh"],2+2*DT*ETA)
assert s.profiles is fork.profiles and s.duration==0
expired = s.fork()
total_missed = 0
for _ in range(36):
    o = expired.step(dict(processing=[5.6,5.6],battery=[0,0]))
    total_missed += o["deadline_miss_mwh"]
assert total_missed>0 and expired.queue.sum()<1e-8
recovered = s.fork()
completed = 0
for _ in range(36):
    o = recovered.step(recovered.default_action("hold"))
    completed += o["completed_mwh"]
assert np.isclose(completed,20*.8*2)
assert recovered.queue.sum()<1e-8
assert all(np.isclose(c["energy_mwh"],2) for c in recovered.campus_view())
print("Storage conservation, workload recovery/expiry and independent forks passed")

# With one charge step left, preserve the energy needed to reach each reserve.
capacity = np.array([10.,10.,10.])
available = np.array([True,True,False])
reserve = np.array([2.,2.,2.])
energy = np.array([2.,2.-2*ETA*DT,1.8])
power = reserve_battery_command([2.,2.,2.],capacity,energy,available,reserve,1)
assert np.allclose(power,[2*ETA**2,0.,0.])
next_energy = energy - np.maximum(power,0)*DT/ETA
recharge = reserve_battery_command([2.,2.,2.],capacity,next_energy,available,reserve,0)
assert np.allclose(recharge,[-2.,-2.,0.])
assert np.allclose((next_energy-recharge*ETA*DT)[:2],reserve[:2])
print("Reserve dispatch bounds passed")
