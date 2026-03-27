---
phase: 20-runtime-integration-and-proof
plan: 02
subsystem: canonical-generation-proof
tags: [paritychecker, generator, runtime, q2_k, q3_k, q6_k]
requires:
  - phase: 20-01
    provides: runtime wrapper accessors for q2/q3/q6 attribution
provides:
  - canonical `paritychecker --generation` publication of q2/q3/q6 runtime attribution
  - active `1/10` generation parity gate aligned to explicit user approval
  - canonical AArch64 proof that supported generation stays on optimized q2/q3/q6 paths
affects: [20-03 regression closeout]
tech-stack:
  added: []
  patterns: [canonical runtime publication, user-approved narrow gate]
key-files:
  created: []
  modified:
    [tools/paritychecker/parity_runner.cpp, tools/paritychecker/paritychecker_tests.cpp]
key-decisions:
  - "Publish q2/q3/q6 runtime attribution from the real quantized GGUF generation path instead of synthetic fixtures."
  - "Narrow the enforced parity gate to `1/10` per explicit user approval while leaving `100/1000` recorded as deferred debt."
requirements-completed: [PAR-04]
duration: 0min
completed: 2026-03-22
---

# Phase 20 Plan 2 Summary

**Canonical generation proof now publishes q2/q3/q6 runtime attribution at the active parity scope**

## Accomplishments

- Updated
  [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp)
  to publish `quantized_dispatch:` metrics for optimized/shared q2/q3/q6 calls and to fail the
  canonical AArch64 proof if any supported q2/q3/q6 path falls back to shared execution.
- Updated
  [paritychecker_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/paritychecker_tests.cpp)
  so the maintained blocking generation parity gate now enforces the active `1/10` lengths and
  checks the new q2/q3/q6 attribution metrics on the canonical fixture.
- Verified directly that
  `./build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 10`
  reports optimized q2/q3/q6 dispatch with zero shared fallback on AArch64.

## Verification

- `cmake --build build/paritychecker_zig --target paritychecker paritychecker_tests -j4`
- `./build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 10`
- `./build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1 --dump`
- `./build/paritychecker_zig/paritychecker_tests --test-case='*active decode lengths*' --no-breaks`

## Deviations from Plan

- The enforced parity length surface is intentionally narrower than the roadmap’s full
  `1/10/100/1000` target because the user explicitly approved treating `1` and `10` as sufficient
  for now. Longer decode parity remains a recorded defer, not a hidden pass.

---
*Phase: 20-runtime-integration-and-proof*
*Completed: 2026-03-22*
