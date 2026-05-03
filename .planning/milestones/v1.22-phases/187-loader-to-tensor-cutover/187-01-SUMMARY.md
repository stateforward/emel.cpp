---
phase: 187-loader-to-tensor-cutover
plan: 01
completed: 2026-05-02
requirements-completed:
  - LOAD-01
  - LOAD-02
  - LOAD-03
  - LOAD-04
---

# Phase 187 Summary

The maintained loader path now uses the tensor-owned `load_tensors` seam and keeps model loading
failure handling explicit and deterministic.

## Evidence

- Renamed loader events, guards, actions, states, and tests from `load_weights` to `load_tensors`.
- Updated generation benchmark, Sortformer fixture, paritychecker, embedded probe, and mock tool
  call sites.
- Paritychecker tests pass after source expectation refresh for the maintained formatting output.
