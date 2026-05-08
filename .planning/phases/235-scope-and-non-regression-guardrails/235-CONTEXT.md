# Phase 235 Context - Scope and Non-Regression Guardrails

## Scope

Phase 235 is guardrail/test/script only for issue #63 / v1.26.  
No runtime source behavior changes are included in this phase work.

Guardrails required:
- GRD-01: detect syscall/file-loop ownership leaks into `model/loader`.
- GRD-02: keep tensor residency ownership in `model/tensor`; prevent migration to loader/io surfaces.
- GRD-03: deny cooperative coroutine staged scheduling scaffolding unless separately approved.
- GRD-04: preserve shipped mmap strategy semantics.
- GRD-05: preserve shipped bulk `io/read` strategy semantics.

## Baseline Patterns Reused

Guardrail style follows existing deterministic source-backed tests plus public `process_event(...)` behavior tests:
- `tests/model/loader/lifecycle_tests.cpp`
- `tests/model/tensor/lifecycle_tests.cpp`
- `tests/io/loader/lifecycle_tests.cpp`
- `tests/io/mmap/lifecycle_tests.cpp`

## Phase 235 Intent

- Broaden ownership/token scans across relevant component headers.
- Keep assertions focused on behavior/ownership boundaries rather than brittle helper-name/comment text checks where possible.
- Validate from milestone worktree only.
