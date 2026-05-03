# Phase 185: Tensor Ownership Contract - Context

**Gathered:** 2026-05-02
**Status:** Complete
**Mode:** Autonomous source inspection

## Phase Boundary

`src/emel/model/tensor` is the canonical owner of tensor load, bind, evict, and residency
semantics. `src/emel/model/loader` remains orchestration over model parse/load phases and must not
own backend-specific tensor residency strategy logic.

## Source Trace

- `src/emel/model/tensor/{events,context,guards,actions,sm}.hpp` owns tensor lifecycle state.
- `src/emel/model/loader/{events,guards,actions,sm}.hpp` owns model load orchestration.
- `src/emel/model/weight_loader/**` was the previous parallel residency owner and is the source of
  the cutover.
- Future `emel/io` strategy work is below tensor ownership and remains deferred.

## Decisions

- Keep strategy selection out of this milestone.
- Preserve public model loader behavior while renaming the long-term seam from `load_weights` to
  `load_tensors`.
- Do not add a transitional adapter unless the maintained path requires it.
