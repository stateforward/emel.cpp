---
phase: 240
plan: 01
status: complete
requirements-completed:
  - CO-02
  - CO-03
  - CO-04
  - CO-05
  - TST-01
---

# Phase 240 Summary

## Completed

- Added `emel::co_sm` as an opt-in wrapper in `src/emel/sm.hpp`.
- Added EMEL policy aliases for coroutine schedulers and allocator policies.
- Added `fixed_coroutine_allocator` with no heap fallback.
- Added contextless and contextful wrapper support with public `process_event`,
  `process_event_async`, `is`, `visit_current_states`, `scheduler`, and `allocator` accessors.
- Added focused `co_sm` doctests to `tests/sm/sm_policy_tests.cpp`.

## Verification

Phase 240 verification passed. The runtime async I/O strategy itself is intentionally deferred to
Phase 241 and later.
