# Grid Shift v2

Grid Shift studies compute placement and operation on a synthetic 36-bus AC benchmark. A proposal couples site capacity with dispatch, storage and immediate/deferred service. The app separates WAM forecasts, heuristic options and AC outcomes.

## Use the application

Open [Grid Shift](http://127.0.0.1:5174/). Set requested MW in Mission, select candidate substations and run the study. Select a proposal, then open a branch to step the simulator. Environment changes revise the scenario and invalidate earlier evidence. The Evidence tab contains the study record and can export JSON.

The default scope uses the intact case, up to four flow-ranked connected outages, and one line-derating stress. The planner records baseline failures and islanding separately. A passing proposal covers the baseline-eligible cases actually evaluated. The broader connected-outage catalogue defines the scenario space.

## Reproduce the run

Run the commands from the project root. The electrical entrypoint expects the previously committed electrical shards in `gridshift-artifacts/gridshift-003/data`. Check previous workspace spend and adjust `--prior-usd` to include outstanding reservations before reserving another job.

```sh
modal run --detach modal_v2.py::pretrain --seconds 420 --prior-usd 1.5
modal run --detach modal_v2.py::generate --chunks 8 --prior-usd 2
modal run --detach modal_v2.py::fit --seconds 420 --prior-usd 3
modal run modal_inference.py::demo
modal run --detach modal_v2.py::evaluate --prior-usd 4
```

Start electrical pretraining and generation independently. Start the resource fit after the warm checkpoint and eight resource shard manifests are committed. To use a finished checkpoint locally, fetch `checkpoint.pt`, `training.json`, and `evaluation.json` from `gridshift-artifacts/gridshift-v2-wam-001` into `artifacts/gridshift-v2-wam-001`. Downloads should write to a temporary path and rename atomically.

To run authenticated Modal studies from the local app, deploy `modal_inference.py`, install Modal in the API environment, and set `GRIDSHIFT_MODAL_STUDIES` and `GRIDSHIFT_MODAL_STEPS` to bounded counts and `GRIDSHIFT_MODAL_UNTIL` to a Unix expiry timestamp. Learned inference runs only in Modal. Without a reservation the local API offers AC simulation and heuristic control. Each remote study reserves a 120-second container timeout and returns its execution backend and GPU.

```sh
PYTHONPATH=. python tests/check_resource_sim.py
python tests/check_v2_api.py --wam --modal --restudy
```

The 30-minute configuration uses effective batch size 8 on the GPU and a progressive subset of the immutable resource shards. The comparison is a bounded experiment. Its recorded sample size and feasibility coverage determine what can be concluded.
