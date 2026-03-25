---
phase: 10-flash-kernel-bring-up
plan: 02
subsystem: kernel-backend-routing
tags: [kernel, flash-attention, backend, workspace, tests]
requires:
  - phase: 10-flash-kernel-bring-up
    provides: the shared canonical `op_flash_attn_ext` implementation landed in Wave 1
provides:
  - backend-owned `flash_attn_workspace` reuse for x86_64 and aarch64 kernel actors
  - real backend dispatch of canonical `op_flash_attn_ext` through the shared EMEL-owned kernel
  - repeated-dispatch unit proof that workspace reuse stays allocation-free and actor-local
affects: [11-generator-flash-adoption, 12-parity-and-verification-closure]
tech-stack:
  added: []
  patterns: [backend-owned persistent workspace, repeated-dispatch reuse proof]
key-files:
  created: []
  modified:
    [src/emel/kernel/detail.hpp, src/emel/kernel/x86_64/actions.hpp, src/emel/kernel/x86_64/context.hpp, src/emel/kernel/aarch64/actions.hpp, src/emel/kernel/aarch64/context.hpp, tests/kernel/lifecycle_tests.cpp, tests/kernel/aarch64_tests.cpp, tests/kernel/x86_64_tests.cpp]
key-decisions:
  - "Keep flash-attention workspace in backend action context so reuse remains actor-owned and allocation-free across dispatches."
  - "Use the existing SML `valid_op_flash_attn_ext` routing rows and add flash-specific action handling instead of duplicating runtime control flow in new guards."
  - "Prove reuse locally with backend action tests on x86_64 and aarch64, leaving generator adoption to Phase 11."
patterns-established:
  - "Backend kernel actors can own reusable scratch buffers in context when reuse must survive across top-level dispatch calls."
  - "Repeated-dispatch proofs should inspect actor-local observable state such as prepared token counts and reuse counters."
requirements-completed: [FLASH-01, FLASH-02]
duration: 0min
completed: 2026-03-21
---

# Phase 10 Plan 2 Summary

**Canonical flash-attention now runs through the real backend actors with persistent workspace reuse proof**

## Accomplishments

- Added actor-owned `flash_attn_workspace` storage to
  [x86_64/context.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/x86_64/context.hpp)
  and
  [aarch64/context.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/context.hpp),
  then routed `op_flash_attn_ext` through backend action execution in
  [x86_64/actions.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/x86_64/actions.hpp)
  and
  [aarch64/actions.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/actions.hpp).
- Extended
  [detail.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/detail.hpp)
  with workspace-backed flash execution so canonical requests reuse a persistent score buffer and
  still reject unsupported request shapes explicitly.
- Added repeated-dispatch proof in
  [lifecycle_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/kernel/lifecycle_tests.cpp)
  and
  [aarch64_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/kernel/aarch64_tests.cpp),
  with supporting context initializer updates in
  [x86_64_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/kernel/x86_64_tests.cpp).

## Task Commits

None - execution is staying local and `commit_docs` is `false`.

## Verification

- `cmake --build build/debug --target emel_tests_bin -j4`
- `ctest --test-dir build/debug --output-on-failure -R emel_tests`
- `scripts/quality_gates.sh`
- `rg 'exec_op_flash_attn_ext|valid_op_flash_attn_ext' src/emel/kernel/x86_64 src/emel/kernel/aarch64`

## Deviations from Plan

- The plan expected backend guard-file edits, but the explicit
  `valid_op_flash_attn_ext` and `invalid_op_flash_attn_ext` SML routing already existed in
  [x86_64/sm.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/x86_64/sm.hpp)
  and
  [aarch64/sm.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/sm.hpp),
  so the required change was action/context wiring plus shared workspace support instead of new
  runtime branching.
- Adding backend-owned workspace to context changed aggregate initialization in existing kernel
  tests, so
  [x86_64_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/kernel/x86_64_tests.cpp)
  and
  [aarch64_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/kernel/aarch64_tests.cpp)
  needed initializer updates alongside the new reuse checks.
- `scripts/quality_gates.sh` again completed with the repo's tolerated benchmark warning for
  existing snapshot drift and the new canonical generation row baseline, but the gate exited
  successfully and no flash-kernel-specific regression blocked the phase.

## Next Readiness

- Phase 11 can now adopt flash attention inside the generator without inventing new backend
  workspace semantics or data-plane helpers.
- Phase 12 can verify flash-path exercise and parity using the real backend actor route instead of
  a tool-local or shared-helper-only shortcut.

---
*Phase: 10-flash-kernel-bring-up*
*Completed: 2026-03-21*
