# Phase 187: Loader-To-Tensor Cutover - Context

**Gathered:** 2026-05-02
**Status:** Complete

## Phase Boundary

Model loader remains responsible for model load orchestration but no longer names the long-term
bulk residency seam as `load_weights`.

## Source Context

- `src/emel/model/loader/events.hpp` exposes `load_tensors_fn`.
- `src/emel/model/loader/{guards,actions,sm}.hpp` use `load_tensors` state and predicate names.
- Maintained bench and parity paths were updated to the tensor-owned naming.
