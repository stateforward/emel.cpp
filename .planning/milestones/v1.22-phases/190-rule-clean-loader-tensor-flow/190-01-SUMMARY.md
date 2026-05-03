---
phase: 190-rule-clean-loader-tensor-flow
plan: 01
completed: 2026-05-02
requirements-completed:
  - TENSOR-02
  - TENSOR-03
  - LOAD-02
  - LOAD-04
---

# Phase 190 Summary

The maintained model-loading path now coordinates tensor-owned bulk residency through the tensor
actor interface instead of a loader callback seam.

## Evidence

- `emel::model::loader::event::load` now carries a tensor actor pointer plus preallocated
  `effect_request` / `effect_result` spans.
- `action::run_load_tensors` drives tensor-owned bind, plan, and apply events and publishes loader
  bytes/weights metadata from the chosen tensor outcome.
- Tensor bulk bind, plan, and apply state-machine rows no longer depend on bulk
  `bind_storage_ctx`, `plan_load_ctx`, or `apply_effect_results_ctx` status fields.
- Generation bench, Sortformer bench, embedded probe, and paritychecker preallocate effect storage
  before loader dispatch.
- Focused loader/tensor tests cover happy path, missing tensor actor/storage, tensor bulk error
  routing, and public wrapper behavior.
