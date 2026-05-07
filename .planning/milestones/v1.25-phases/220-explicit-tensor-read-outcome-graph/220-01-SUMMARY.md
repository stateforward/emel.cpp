---
phase: 220-explicit-tensor-read-outcome-graph
plan: 01
status: complete
completed: 2026-05-05T21:35:00Z
requirements:
  - TIO-02
---

# Phase 220 Summary

## Completed

Phase 220 repaired the tensor read/copy outcome graph so `model/tensor`
selects read-backed tensor outcomes from an explicit same-RTC read result
carrier instead of callback-mutated tensor-local success/error fields.

## Source Changes

- Added `emel::io::read::events::read_tensor_result` as the same-RTC result
  carrier for accepted, success, copied-byte, target-buffer, and read error
  data.
- Added an `io/read::sm::process_event(read_tensor, read_tensor_result&)`
  overload that captures the read actor's completed status while preserving the
  existing public callback-based `read_tensor` API.
- Updated `model/tensor` read/copy dispatch to call the read result overload
  and route success, unsupported, validation failure, file-open failure, and
  file-read failure through explicit guards in `guards.hpp` and transitions in
  `sm.hpp`.
- Removed the prior tensor read outcome callback thunk pattern and the mirrored
  `io_read_ok` / `io_read_err` status fields from the tensor read-load status.
- Added public doctest coverage and source guardrails for representative read
  outcomes and callback-mediated outcome removal.

## Notes

This phase keeps public tensor callbacks as immediate same-RTC replies. Those
callbacks no longer decide which read-backed tensor outcome path runs.

