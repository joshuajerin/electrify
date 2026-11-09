# Connection mission

The running app at http://127.0.0.1:5174 has a connection-study workflow in the right inspector. The scene retains its existing top-down, approximately 80% desktop layout.

## Demo flow

1. Set requested compute in the simulation controls.
2. Select **Find a connection plan**. Progress reports actual AC work.
3. Inspect original-request failures. An overload's **Locate** button selects the limiting line in the scene and inspector.
4. Compare the WAM cap proposal with explicit relocation and smaller-connection alternatives.
5. Change a selected line rating or background demand. With re-study enabled, a new revision-bound study starts automatically. Earlier evidence is marked stale.
6. Select **Explore this branch** to create an independent session with the selected bus and admitted MW. The requested MW stays unchanged, so curtailment remains visible. Ordinary forecasts, stepping and replay then work on this branch.
7. Export the complete JSON study, including all cases and unresolved failures.

## Implemented verification

- Intact network and every currently active branch outage, including outages that the operator-action catalogue would prohibit as intentional switching.
- Baseline with zero new compute, paired with every proposal's cases.
- Separate counts for newly failing cases, unresolved cases, and measured worsening of baseline failures in peak loading, lost existing demand or solver failure.
- Three candidate buses, 25/50/75/100% request alternatives, plus a WAM cap proposal when a checkpoint is loaded.
- An asynchronous single-worker study queue with progress and supersession cancellation within a session.
- Server-side revision checking when opening proposal branches.

The recommendation first chooses the highest-MW proposal passing all cases. If none passes, it may identify an exploratory option with no new failures and no measured worsening. This is explicitly labeled and is not unconditional hosting capacity. Baseline problems remain visible.

## Model boundary

The existing WAM imagines cap actions over six five-minute steps. Its chosen cap and predicted trajectory are stored separately from current-point AC outage verification. Relocation and the other discrete alternatives are explicit engineering search, not learned planning. The local smoke checkpoint is a plumbing demonstration; this feature does not establish generalization or learned adaptation.

The next model extension would learn spatial allocation proposals and evaluate time-coupled feasibility. That requires new training labels and evaluation; it has not been implemented here. Multi-site simultaneous compute allocation, batteries, seasonal firm-capacity claims, real utility network reconstruction, protection dynamics and utility approval are outside this version.

## Validation on the local smoke network

Live HTTP integration: 13 proposals, 59 active branch outages plus the intact case, 840 AC solves, about 24 seconds in the observed run. Verified WAM provenance, case coverage, independent proposal branches, retained requested MW, unchanged parent session and HTTP 409 for a stale study. Ten baseline cases failed and remained in the report.

Evidence: `artifacts/local-smoke/mission-api-check.json`. Frontend TypeScript and production build passed. The browser exercised the mission UI; its headless SwiftShader WebGL context could not initialize, so this run does not validate the 3D rendering visually.
