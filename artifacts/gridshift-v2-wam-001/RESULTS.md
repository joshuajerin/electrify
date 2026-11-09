# Grid Shift v2 experiment

The main 57,964,080-parameter checkpoint completed 182 resource training updates on an NVIDIA L4 in 422 seconds. Electrical pretraining completed 314 updates on the L4 before resource training. The committed resource dataset has 768 windows, 9,216 transitions and 379 source episodes.

- [Resource training run](https://modal.com/apps/joshuajerin/main/ap-AN7v9rpvwEfwBaHOjUMcdE), container `ta-01M6YSD7D9S428HYBKQY8NSFMR`.
- [Electrical pretraining run](https://modal.com/apps/joshuajerin/main/ap-BC3QKGLMmOW5IvePzZ1UdL).
- [Held-out evaluation and failure refinement](https://modal.com/apps/joshuajerin/main/ap-40vyWDBxnXHUUkClz2KOSZ).
- [Authenticated Modal inference deployment](https://modal.com/apps/joshuajerin/main/deployed/gridshift-inference-v2).

## Workflow verification

The browser and the HTTP integration test exercised a real Modal study, the WAM forecast and separate AC trajectory, proposal branching, a learned step executed on Modal, replay, and a fresh study after a rating change. The API rejected a proposal from the earlier revision. The local API runs the AC simulator and sends learned requests to Modal.

In the recorded 140 MW study, the single-site and sensitivity-ranked split proposals overloaded equipment. The WAM proposal failed the terminal battery requirement. The explicit witness-guided repair passed for 112 MW across the baseline-feasible cases. Two of six tested cases already failed without new load. The conditional result applies to the four baseline-feasible cases.

Native benchmark parity passed at 100 points, with maximum flow error of 0.00001083 MW. Battery conservation, deferred-work expiry and recovery, independent forks, local tiny-batch learning, and the frontend production build passed.

## Initial held-out comparison

The wall-clock limit allowed 11 of the 24 fixed-suite episodes. Eight had a feasible zero-addition baseline.

| Controller | Mean verified MW | Mean seconds per replay |
|---|---:|---:|
| Workload-aware heuristic | 26.25 | 0.56 |
| AC candidate search | 35.00 | 4.27 |
| Policy-only ablation | 35.00 | 2.47 |
| WAM imagination with fixed allocation | 0.00 | 3.99 |
| WAM allocation and imagination | 0.00 | 3.94 |

The mean verified MW counts a failed proposal as zero, even when it served all arriving work. The policy-only ablation uses the same checkpoint without imagined ranking. The AC baseline uses short causal candidate search. On four held-out forecast windows, persistence had lower mean loading error than the WAM at 5, 15, 30 and 60 minutes. These results do not establish that the world model improves grid capacity planning.

See `evaluation.json` for the raw records, transfer checks, and refinement results. See `workflow-replay.json` for the two revision-bound Modal studies.

## Failure refinement

Six training-only failure windows were collected with the policy head and relabeled by the AC teacher. The separate checkpoint completed 39 refinement updates on Modal. On 8 paired baseline-feasible cases, mean verified capacity with WAM imagination rose from 0 to 26.25 MW. The paired bootstrap interval was 13.12 to 35.00 MW. The policy-only ablation and AC baseline both averaged 35 MW. The UI's tested deployment uses the initial checkpoint. The refined weights remain in Modal at gridshift-v2-wam-001/refined/checkpoint.pt.

## Focused stress replays

A separate [Modal run](https://modal.com/apps/joshuajerin/main/ap-4zsezzpHx24PS85542TdMp) compared both checkpoints and two baselines at the same 35 MW split on held-out episode 517. All six zero-added-load baselines were feasible. Both WAM rollout modes passed with batteries at minimum charge and with batteries unavailable. Both failed the intact, forced-deferral, hidden demand increase, and connected outage cases on the terminal battery reserve. The policy-only ablation and workload-aware heuristic passed all six. These are paired perturbations of one episode. The full traces, controller actions, and violation witnesses are in `edge-cases.json`.

The planner now bounds battery dispatch at each simulated step using the charge rate, efficiency, and steps remaining before the terminal reserve deadline. The first action sent to the simulator uses the same constrained battery command that the rollout ranked. Both checkpoints and the AC simulator use their existing weights and dynamics.

- The six focused stress replays went from 2/6 to 6/6 passes for each checkpoint. In the forced-deferral replay, the deployed checkpoint completed 56 MWh, missed zero deadlines, and finished at the 7.00 MWh reserve instead of 4.30 MWh. [Modal stress rerun](https://modal.com/apps/joshuajerin/main/ap-9tmfg2yTRdCtlcpyzaa1ib)
- On the same fixed placements in the broader replay, both checkpoints passed all eight baseline-feasible cases at 35 MW with zero violations. The deployed checkpoint previously averaged 0 MW verified, and the refined checkpoint averaged 26.25 MW verified. The policy-only and AC search baselines also verified 35 MW. Three cases had infeasible zero-addition baselines. [Modal held-out rerun](https://modal.com/apps/joshuajerin/main/ap-T27eq2tcZFbk1tD0StEW8u)
- After the hidden demand increase, the deployed checkpoint's one-step mean loading error was 7.33 percentage points, compared with 0.61 for carrying forward the last reading. The reserve change improves dispatch feasibility. The forecast remains weaker than persistence on that replay.

The deployed `battery-reserve-v1` planner passed the live 60 MW HTTP workflow. The test verified Modal provenance, proposal branching, a learned control step, replay history, and rejection of a stale proposal after an edit. See `workflow-reserve.json`.

## Fresh held-out load and disturbance sweep

The reserve-constrained planner was tested on six additional held-out episodes at 35, 70, and 105 MW. The 18 scenarios balanced intact operation, a connected branch outage, and an outage combined with hidden demand growth and low starting charge. Eleven had a feasible zero-added-load baseline. Across those cases the deployed WAM passed 10/11, including 4/4 at 35 MW, 4/4 at 70 MW, and 2/3 at 105 MW. The refined WAM, policy head alone, and workload-aware heuristic passed 11/11. The learned placement and dispatch path passed 10/11 at the full request. [Modal sweep](https://modal.com/apps/joshuajerin/main/ap-oyfvsBQKb6gXw1GqveE4ad)

The failure at 105 MW was on episode 243 after branch 13 opened. At step 27 the deployed WAM requested 21 MW of battery discharge with zero compute processing. Generator 21 fell to −0.174 MW against a 0 MW lower limit. Replaying the same state and action with battery discharge disabled cleared the violation. The learned placement path had two analogous violations and the same one-step counterfactual cleared each. These counterfactuals verify individual steps. [Modal diagnostic](https://modal.com/apps/joshuajerin/main/ap-pMNFOc5vX0K67cQTDbre6C)

Over 396 one-step forecasts in the baseline-feasible cases, the deployed WAM averaged 6.49 percentage points of branch-loading error versus 1.03 for persistence. The refined WAM averaged 7.00 versus 1.06 on its own action traces. No actual branch overload occurred in those eligible WAM traces, so this sweep does not measure overload detection recall. Raw traces are in `model-sweep.json` and `model-sweep-discharge.json`.

## Flexible workload repair and paired grading

The planner now tries flexible processing at the same location and MW request when immediate processing fails a thermal check. Each repair records the proposal it revises and the before-and-after failed-case counts. Its AC probes compare full processing and two levels of deferral for the next declared simulation interval. Interactive replay can run the same policy. The existing deadline queue protects immediate work and counts missed deferred work explicitly.

The paired experiment used 12 held-out demand profiles and constructed a 15-minute derating at the same synthetic corridor, line 54 serving bus 7. The derating for each profile was set between the zero-addition and immediate-processing flows before running the flexible policy. Immediate processing passed **0/12** at 105 MW. The automatically proposed flexible repair passed **12/12** at the same site and request, completing **168 MWh** per replay with zero missed deadlines, existing load loss, or battery dispatch. The sustained version of the same derating failed **12/12** with flexibility even though the zero-addition baseline passed all twelve. These cases isolate the mechanism under declared schedules; they do not estimate how frequently the condition occurs in operation. [Paired Modal run](https://modal.com/apps/joshuajerin/main/ap-XqxDMr04nGYJzAJUR9mOes)

On episode 420, the fixed-processing sweep passed at **85 MW** and failed at **90, 95, 100, and 105 MW**. The flexible plan passed at **105 MW**, 20 MW above the highest passing fixed request tested. At 105 MW, the original proposal reached **122.1%** of the derated line limit. The repair deferred up to **6.3 MWh** and cleared it without missed deadlines. [Capacity sweep](https://modal.com/apps/joshuajerin/main/ap-f7HVsU1XTpL6Cdngn2Mrsc)

The structured proposals and grades are in `flexible-workload-study.json` and the tested capacity levels are in `flexible-capacity-bound.json`. The gains come from AC-verified operational search and workload scheduling.

The live 180 MW study exercised all six proposals and replayed the flexible branch through the API. It rejected the full request, then recorded a reduction from four failed cases to one on the 144 MW proposal, which remained rejected. A separate replay check confirmed that a successful repair opens on its parent's failing scenario. [Repair replay check](https://modal.com/apps/joshuajerin/main/ap-QmzGIbtO1WMzscQCx7vhQU)
