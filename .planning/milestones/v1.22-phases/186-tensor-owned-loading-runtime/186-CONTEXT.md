# Phase 186: Tensor-Owned Loading Runtime - Context

**Gathered:** 2026-05-02
**Status:** Complete
**Mode:** Autonomous implementation

## Phase Boundary

Add tensor-owned bulk residency events, guards, actions, context, and transitions while preserving
existing per-tensor bind, evict, and capture-state behavior.

## Source Context

- `src/emel/model/tensor/events.hpp` carries public bulk events and explicit outcome events.
- `src/emel/model/tensor/context.hpp` stores actor-owned bound tensor records only.
- `src/emel/model/tensor/sm.hpp` owns explicit load, plan, and apply phases.
- `tests/model/tensor/lifecycle_tests.cpp` covers both existing lifecycle and new bulk behavior.
