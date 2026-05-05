---
phase: 204-mmap-strategy-component-boundary
plan: 01
status: validated
requirements:
  - MMAP-01
created: 2026-05-04T15:22:00Z
last_updated: 2026-05-04T16:12:00Z
---

# Phase 204 Plan 01 Summary

## Implementation

- Added `src/emel/io/mmap/{context,errors,events,detail,guards,actions,sm}.hpp` as the canonical
  Stateforward.SML mmap strategy actor skeleton, mirroring the layout of
  `src/emel/io/loader`.
- Modeled the boundary as a fail-closed actor with named states
  `state_ready`, `state_request_decision`, `state_invalid_request_error_decision`,
  `state_unsupported_platform_error_decision`, and `state_error_callback`.
  All accepted requests route to `state_unsupported_platform_error_decision` because Phase 204
  intentionally does not implement validation, mapping, or descriptors. Validation, mapping
  attempts, and descriptor publication are explicitly deferred to Phases 205-206.
- Public event surface: `event::map_tensor_request` (file/offset/length identity), `event::map_tensor`
  with required-by-reference request and optional `on_done`/`on_error` callbacks.
  `events::map_tensor_done` and `events::map_tensor_error` form the publication contract.
- Component-local `action::context` is empty; no dispatch-local data, no tensor residency, no
  request mirroring. The `detail::map_tensor_runtime` carrier bridges public requests to internal
  completion progress without copying request payload into context.
- Public alias `emel::io::mmap::sm` is exposed via `src/emel/io/mmap/sm.hpp` and the additive
  top-level alias `emel::IoMmap` is added in `src/emel/machines.hpp`. `emel::io::sm` remains
  bound to `emel::io::loader::sm`; tensor-side and runtime exposure is deferred to Phases
  207-208.
- Doctest coverage in `tests/io/mmap/lifecycle_tests.cpp` drives the actor only through
  `process_event(...)` and `is(...)`/`visit_current_states`. Cases cover canonical alias visibility,
  invalid-request fail-closed, unsupported-platform fail-closed, recovery to `state_ready`,
  callback-absent fallthrough, unexpected-event handling, and a source-text scope guardrail.
- Generated architecture docs at `.planning/architecture/io_mmap.md` and the mermaid baseline at
  `.planning/architecture/mermaid/io_mmap.mmd` were produced by the maintained `generate_docs`
  flow (CMake `generate_docs` test) and reflect the new actor.

## Boundary Discipline

- No `mmap`/`munmap`/`CreateFileMapping`/`MapViewOfFile`/`pread`/`std::ifstream` calls in
  `actions.hpp` or `detail.hpp`; the boundary actor performs no platform IO.
- No staged read/copy, device-specific, cooperative async, model-family widening, or tool-only
  mmap scaffold lives in `src/emel/io/mmap`.
- `model/tensor` is unchanged: tensor residency lifecycle ownership is preserved.
- `model/loader` is unchanged: orchestration-only contract is preserved.

## Validation Evidence

- `cmake --build build/zig --target emel_tests_bin` — succeeded.
- `build/zig/emel_tests_bin --no-breaks --source-file=*tests/io/mmap/lifecycle_tests.cpp` —
  7 cases, 40 assertions, 0 failed.
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_io` — `emel_tests_io` passed.
- `scripts/check_domain_boundaries.sh` — exit 0.
- `scripts/lint_snapshot.sh` — exit 0 (no regressions; new files clang-formatted).
- `EMEL_QUALITY_GATES_CHANGED_FILES="...mmap files..." scripts/quality_gates.sh` —
  - `lint_snapshot`: passed.
  - `test_with_coverage` (scoped to `src/emel/io/mmap/...` and `src/emel/machines.hpp`): line
    coverage 94.3% (>=90% threshold), branch 0.0%/0 (no branches in scoped surface), functions
    91.7%.
  - `paritychecker`, `fuzz_smoke`: skipped — no affecting changed files.
  - `bench_snapshot`: status=1 due to broad-src bench triggering on `src/emel/machines.hpp`. The
    failing benchmarks (`tokenizer/preprocessor_rwkv_long`, `text/encoders/rwkv_long`,
    logits/sampler, validator, batch/planner_simple in another run) are pre-existing on this
    branch — verified by stashing the Phase 204 changes and rerunning the same gate, which still
    surfaces benchmark regressions in unrelated suites with similar magnitude. Phase 204 changes
    do not touch `src/emel/text`, `src/emel/logits`, or `src/emel/batch` runtime code paths.

## Outstanding

- Resolved on 2026-05-04: main approved option (a), use
  `EMEL_QUALITY_GATES_ALLOW_BENCH_REGRESSION=1` for the Phase 204 transitional gate only.
  Final transitional gate run completed with all lanes green except `bench_snapshot` which
  was ignored by the explicit override (see `204-VALIDATION.md`). Phase 210 must enforce
  final benchmark/publication truth across the maintained runtime/parity/docs paths and
  remove the transitional override before milestone v1.24 closeout.
