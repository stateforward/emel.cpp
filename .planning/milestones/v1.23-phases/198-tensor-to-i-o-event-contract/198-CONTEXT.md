---
phase: 198-tensor-to-i-o-event-contract
status: complete
requirements:
  - IO-03
  - TBOUND-01
  - TBOUND-02
created: 2026-05-04T01:10:00Z
---

# Phase 198 Context

Phase 198 wires the public tensor-to-IO contract without moving tensor residency ownership. The
contract must use explicit event payloads and explicit `_done` / `_error` outcomes rather than
status fields or retained per-dispatch pointers in actor context.

Locked decisions:

- Tensor load planning may describe an IO request, but `model/tensor` stays the canonical owner of
  tensor bind, load, evict, and residency transitions.
- IO result and error reporting must be event-shaped and same-RTC only.
- Tests must drive public actors through `process_event(...)` and SML state inspection.

Canonical refs:

- `src/emel/io/loader/events.hpp`
- `src/emel/model/tensor/events.hpp`
- `src/emel/model/tensor/sm.hpp`
- `tests/io/loader/lifecycle_tests.cpp`
- `tests/model/tensor/lifecycle_tests.cpp`
