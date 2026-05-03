---
phase: 190
slug: rule-clean-loader-tensor-flow
status: complete
created: 2026-05-02
---

# Phase 190 Context

Milestone audit found that the maintained loader-to-tensor path still routed tensor residency work
through a loader `load_tensors` callback. Maintained tools allocated effect storage inside that
callback and then called `tensor_loader.process_event(...)` from the callback body during loader
dispatch.

The phase closes that gap by making `model/loader` coordinate a tensor-owned actor directly through
pre-bound public tensor bulk events:

- `event::bind_storage`
- `event::plan_load`
- `event::apply_effect_results`

Maintained tools must allocate their tensor effect request/result storage before the loader dispatch
starts and pass spans into `event::load`.
