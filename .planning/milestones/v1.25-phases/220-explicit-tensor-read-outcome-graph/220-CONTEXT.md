---
phase: 220-explicit-tensor-read-outcome-graph
status: ready
created: 2026-05-05T20:50:00Z
requirements:
  - TIO-02
depends_on:
  - 219-maintained-read-source-provenance
---

# Phase 220 Context

## Goal

Close the tensor read outcome graph gap: `model/tensor` must select read success,
unsupported, validation failure, file open failure, and file read failure through
explicit typed same-RTC result data and guarded transitions, not through
callbacks that mutate tensor-local status later inspected by guards.

## Ordering Reconciliation

Some Phase 220 source edits were started before this context and plan existed
while repairing Phase 219. This artifact records that ordering mistake and
keeps the Phase 220 work distinct from Phase 219 source-provenance work.

Already-started Phase 220 files:

- `src/emel/io/read/detail.hpp`
- `src/emel/io/read/events.hpp`
- `src/emel/io/read/sm.hpp`
- `src/emel/model/tensor/actions.hpp`
- `src/emel/model/tensor/detail.hpp`
- `src/emel/model/tensor/guards.hpp`
- `tests/io/read/lifecycle_tests.cpp`
- `tests/model/tensor/lifecycle_tests.cpp`

## Source-Backed Truth

- `io/read` already publishes explicit `read_tensor_done` and
  `read_tensor_error` events through public callbacks for normal callers.
- `model/tensor` needs a same-RTC outcome capture that does not use those
  callbacks to decide which tensor transition path runs.
- The tensor state machine must keep behavior selection in `guards.hpp` and
  `sm.hpp`; actions may execute the already-selected dispatch and copy result
  fields, but must not choose success/error paths.
- Public doctests must drive outcomes through `process_event(...)` and inspect
  state/results.

## Decisions

- Add an `io/read` `read_tensor_result` result carrier and overload that captures
  final same-RTC status directly from the read actor's internal attempt status.
- Have `model/tensor` dispatch `io/read` with this result carrier instead of
  installing read callbacks that mutate `request_read_load_status`.
- Keep tensor outcome guards explicit over the typed result fields.
- Preserve public callback behavior for callers that still use normal
  `io/read::event::read_tensor`.

