# Grid Shift · judging evidence

**Track 2 question.** For a new compute load at a substation, how much power can the operator admit while meeting modeled equipment and voltage limits, and how does that answer change when a line is unavailable? This addresses RFQ 2.1 (substation headroom) and RFQ 2.3 (single-outage sensitivity). [Organizer problem list](https://raw.githubusercontent.com/arhamshah08/energy-hack/main/ai/track-2-grid-capacity.md)

| Criterion | What to show |
|---|---|
| Agentic loop | Start with immediate processing. A thermal violation triggers a same-site, same-MW flexible proposal. The agent reruns the scenario suite and records the parent proposal, grade, and measured change. If that repair fails, it can try moving the load or reducing the request. |
| Energy-system innovation | Compare immediate processing and deadline-aware deferral on the same campus and line. In the measured temporary derating, the highest passing fixed request tested was 85 MW. The flexible policy admitted 105 MW and cleared all work without battery dispatch. |
| Evaluation and learning | Compare zero-addition, immediate-processing, flexible-processing, and sustained-derating runs. Across 12 held-out demand profiles with constructed temporary deratings, the automated update changed the grade from 0/12 to 12/12 at 105 MW. The sustained cases all failed, establishing a limit on what deferral can recover. |
| Technical execution | Drag the compute load, trip a line, compare forecast and simulated outcomes, and fork the same state to replay another action. AC parity was checked at 100 local operating points. |
| Theme alignment | Offer an operating limit tied to the specific equipment and scenario rather than a grid-wide supply estimate. This is an input to a utility study. |

## Measured local run

The initial request was **69.6 MW**. At bus 0, line 2 reached **118.5%** of its rating. Feedback narrowed the feasible cap to **58.72 MW** within seven bisection trials. Bus 32 passed at **69.6 MW**, the search ceiling, for the intact case and all four selected outages. That is **10.88 MW** more than the best tested cap at bus 0. The study reports three separately examined outages that already failed at zero incremental compute. The selected outage set and complete attempts are in `artifacts/local-smoke/track2-capacity.json`.

The model checkpoint served in the local demo has **51,426,068 parameters** and was trained on a tiny batch. It demonstrates integration. The capacity study numbers come from AC simulation and feedback search. They do not measure learned generalization.
