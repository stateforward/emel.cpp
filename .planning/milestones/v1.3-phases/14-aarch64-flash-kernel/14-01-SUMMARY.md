---
phase: 14-aarch64-flash-kernel
plan: 01
subsystem: kernel-aarch64-flash
tags: [kernel, aarch64, flash-attention, neon, tests]
requires:
  - phase: 13-benchmark-evidence
    provides: the shipped canonical flash-backed generation slice from v1.2
provides:
  - a backend-specific AArch64 `op_flash_attn_ext` fast path for the maintained canonical request
  - backend context observability for optimized vs shared flash dispatches
  - failing-then-passing AArch64 regression coverage for the optimized flash path
affects: [14-02 proof closure, 15-runtime-adoption-and-proof]
tech-stack:
  added: []
  patterns: [aarch64 online softmax accumulation, backend-owned flash path accounting]
key-files:
  created: []
  modified:
    [src/emel/kernel/aarch64/actions.hpp, src/emel/kernel/aarch64/context.hpp, tests/kernel/aarch64_tests.cpp]
key-decisions:
  - "Keep the shared flash validator in `src/emel/kernel/detail.hpp` as the truth gate and replace only the AArch64 data plane."
  - "Use in-place online softmax accumulation over the destination buffer so the optimized path stays allocation-free."
  - "Expose optimized vs shared dispatch counters in backend context so the proof surface stays kernel-local."
patterns-established:
  - "Backend-specific flash optimization can land inside the existing AArch64 action path without changing SML rows."
  - "AArch64 flash-path proof should be observable through persistent backend counters rather than tool-only instrumentation."
requirements-completed: []
duration: 0min
completed: 2026-03-22
---

# Phase 14 Plan 1 Summary

**The maintained AArch64 flash request now has a native backend execution path**

## Accomplishments

- Added a failing AArch64 regression first in
  [aarch64_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/kernel/aarch64_tests.cpp)
  by asserting that canonical flash dispatch must increment optimized-path accounting on the
  backend context.
- Extended
  [actions.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/actions.hpp)
  with an AArch64-native `op_flash_attn_ext` helper that uses NEON row math plus online softmax
  accumulation over the existing strided K/V cache layout.
- Added persistent observability to
  [context.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/context.hpp)
  so the backend records optimized vs shared flash dispatch counts across calls.

## Task Commits

None - execution is staying local and `commit_docs` is `false`.

## Verification

- `cmake --build build/zig --target emel_tests_bin --parallel 8`
- `./build/zig/emel_tests_bin --test-case='*aarch64*flash_attn_ext*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='kernel_*' --no-breaks`

## Deviations from Plan

- The optimized path did not require SML guard or transition changes; the existing
  `exec_op_flash_attn_ext` specialization in
  [actions.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/aarch64/actions.hpp)
  was sufficient.
- The shared scalar helper remains present as the non-optimized fallback for valid requests
  outside the AArch64 fast path; Plan 1 only needed to guarantee that the maintained canonical
  request no longer routes through it on ARM.

## Next Readiness

- Plan 2 can focus entirely on correctness, unsupported-path truth, reusable scratch proof, and
  repo-wide validation instead of backend bring-up.
- Phase 15 can consume the optimized AArch64 path through the existing runtime chain once kernel-
  local proof is closed.

---
*Phase: 14-aarch64-flash-kernel*
*Completed: 2026-03-22*
