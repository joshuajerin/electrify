# Electrify

Electrify is a power-grid connection study and 3D simulator built at SF × MIT Hack Day: Energy. It proposes where to connect data-center load, checks those plans in an AC simulator, and uses the recorded failure to try a revised operating schedule or placement.

[Live demo](https://electrify-demo.vercel.app) · [Evaluation report](artifacts/gridshift-v2-wam-001/RESULTS.md) · [Training and inference runbook](V2-RUNBOOK.md)

## Run locally

With Python 3.11, uv, and Node.js installed, run the API from the repository root:

```sh
uv venv --python 3.11
uv pip install -r requirements-modal.txt
GRIDSHIFT_SMOKE=1 .venv/bin/python -m uvicorn gridshift.api:app --port 8000
```

In a second terminal, run the frontend:

```sh
cd ui
npm ci
npm run dev
```

Open [localhost:5174](http://localhost:5174). Choose the benchmark grid, set the requested megawatts, and run a study. Compare proposals and open a branch to replay the result or try a different operating policy. This setup uses the packaged Grid2Op test profiles, AC simulation, and heuristic controls. The runbook covers the larger benchmark and authenticated Modal inference.

To verify the simulator locally:

```sh
PYTHONPATH=. .venv/bin/python tests/check_resource_sim.py
```

## World Action Model

The PyTorch model uses five graph message-passing blocks and separate 768-dimensional GRU states for buses and branches. The 57,964,080-parameter checkpoint takes 12 five-minute observations and past actions. It scores placements and explicit operating commands, and predicts up to an hour of bus voltages, branch flows, and service outcomes. The planner ranks candidate schedules using those forecasts and checks them separately with the AC simulator. Model weights and generated training datasets are stored separately from this repository.

## What the evals showed

On the synthetic benchmark, immediate processing at 105 MW failed all 12 scenarios with a constructed temporary line derating. The AC-guided repair passed all 12 at the same site and capacity by deferring eligible work, with zero missed deadlines or existing load loss. It failed all 12 when the derating was sustained.

The learned model's average one-step branch-loading error was 6.49 percentage points across 396 forecasts, versus 1.03 for carrying forward the last reading. The held-out control sweep also favored the simpler policy-only baseline. The measured capacity improvement came from workload scheduling and AC verification. See the [evaluation report](artifacts/gridshift-v2-wam-001/RESULTS.md) for scenario selection, comparisons, and limitations.
