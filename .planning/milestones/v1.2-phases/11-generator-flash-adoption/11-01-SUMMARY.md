---
phase: 11-generator-flash-adoption
plan: 01
subsystem: generator-flash-routing
tags: [generator, flash-attention, kernel, observability, tests]
requires:
  - phase: 10-flash-kernel-bring-up
    provides: backend-owned canonical `op_flash_attn_ext` execution in `src/emel/kernel`
provides:
  - canonical generator dispatch of `op_flash_attn_ext` through the real kernel actor
  - generator-owned flash-specific observability separate from generic kernel dispatch counts
  - canonical flash-request shape proof over the existing position-major K/V cache layout
affects: [12-parity-and-verification-closure]
tech-stack:
  added: []
  patterns: [generator-owned flash observability, truthful position-major flash request formation]
key-files:
  created: []
  modified:
    [src/emel/generator/detail.hpp, src/emel/generator/sm.hpp, src/emel/kernel/detail.hpp, tests/generator/detail_tests.cpp, tests/generator/lifecycle_tests.cpp]
key-decisions:
  - "Route canonical generation through `op_flash_attn_ext` instead of retaining a generator-local materialized attention seam."
  - "Accept the generator's existing position-major K/V cache views as canonical flash operands rather than forcing a fake dense-contiguous layout."
  - "Expose generator-owned flash dispatch counts so later parity work can prove flash execution without conflating it with generic kernel traffic."
patterns-established:
  - "Generator/runtime proof surfaces live in generator-owned observability, not in tool-local counters."
  - "Canonical flash request formation may preserve truthful strided K/V cache layout when the kernel validator explicitly models that layout."
requirements-completed: [GEN-01]
duration: 0min
completed: 2026-03-21
---

# Phase 11 Plan 1 Summary

**Canonical generation now dispatches the real flash-attention kernel and records flash-specific execution**

## Accomplishments

- Added generator-local flash request construction and dispatch in
  [detail.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/generator/detail.hpp),
  so the active `run_layer(...)` path now routes canonical attention through
  `emel::kernel::event::op_flash_attn_ext` over `backend.q`, `backend.key_cache`,
  `backend.value_cache`, and `backend.attn_ctx`.
- Relaxed canonical flash operand validation in
  [detail.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/detail.hpp)
  so the generator's truthful position-major K/V cache strides are accepted without pretending K/V
  are dense-contiguous tensors.
- Added flash-specific generator observability in
  [detail.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/generator/detail.hpp)
  and surfaced it from
  [sm.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/generator/sm.hpp)
  via `generation_flash_attention_dispatch_calls()`.
- Extended
  [detail_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/generator/detail_tests.cpp)
  and
  [lifecycle_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/generator/lifecycle_tests.cpp)
  to prove canonical flash request shape and generator-level flash dispatch visibility.

## Task Commits

None - execution is staying local and `commit_docs` is `false`.

## Verification

- `cmake --build build/debug --target emel_tests_bin -j4`
- `./build/debug/emel_tests_bin --test-case='generator_detail_builds_flash_request_over_position_major_kv_cache' --no-breaks --force-colors=0`
- `./build/debug/emel_tests_bin --test-case='*generator*' --no-breaks --force-colors=0`
- `rg 'op_flash_attn_ext|flash_attention_dispatch_calls|generation_flash_attention_dispatch_calls' src/emel/generator src/emel/kernel tests/generator`

## Deviations from Plan

- Generator lifecycle verification exposed a test-only stack overflow unrelated to flash routing:
  [lifecycle_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/generator/lifecycle_tests.cpp)
  had a heap fixture that still materialized `prepared_model`-sized temporaries during
  construction. The fixture was rewritten to build the large model in place and keep the
  generator machine heap-owned so the native runtime path could be tested safely under debug and
  coverage builds.
- The compliance checklist forced one small code-path adjustment in
  [detail.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/generator/detail.hpp):
  flash dispatch accounting uses branchless arithmetic instead of a runtime `if` in the detail
  call path.

## Next Readiness

- Phase 11 Plan 2 can now prove canonical success and deterministic unsupported-request failure
  using the generator-owned flash observability added here.
- Phase 12 can consume the new generator-local flash counters when publishing truthful parity and
  verification proof.

---
*Phase: 11-generator-flash-adoption*
*Completed: 2026-03-21*
