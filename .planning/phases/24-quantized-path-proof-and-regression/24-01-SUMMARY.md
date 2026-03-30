---
phase: 24-quantized-path-proof-and-regression
plan: 01
subsystem: paritychecker-runtime-proof
tags: [paritychecker, quantized, runtime, proof, regression]
requires:
  - phase: 23-arm-quantized-path-closure
    provides: shipped generator runtime contract counts and canonical `8/4/0/0` proof
provides:
  - maintained paritychecker publication of the shipped runtime contract counts
  - canonical generation failure semantics for disallowed fallback or no-claim regressions
  - exact maintained `1/10/100/1000` parity assertions over the approved contract
affects: [24-02 generator regression hardening]
tech-stack:
  added: []
  patterns: [runtime-contract publication, parity hard-fail proof, audit-to-runtime consistency]
key-files:
  created: []
  modified:
    [tools/paritychecker/parity_runner.cpp, tools/paritychecker/paritychecker_tests.cpp]
key-decisions:
  - "Use the shipped generator wrapper as the primary proof source and keep the model-audit inventory as a consistency surface, not the other way around."
  - "Fail canonical generation proof if `disallowed_fallback` or `explicit_no_claim` appears, while keeping approved dense-f32-by-contract stages visible."
requirements-completed: [ATTR-01, PAR-05]
duration: 0min
completed: 2026-03-25
---

# Phase 24 Plan 1 Summary

**Maintained paritychecker proof now consumes the shipped runtime contract**

## Accomplishments

- Extended
  [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp)
  so generation mode now publishes `quantized_runtime_contract:` from the shipped generator wrapper
  before the existing stage inventory rows.
- Hardened canonical generation proof so paritychecker now fails if the supported path reports any
  `disallowed_fallback` or `explicit_no_claim` stages, and also fails if the shipped runtime
  counts drift from the shared model-audit inventory.
- Tightened
  [paritychecker_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/paritychecker_tests.cpp)
  so the maintained `1/10/100/1000` loop now asserts the exact `8/4/0/0` runtime contract instead
  of only checking that audit strings exist.

## Verification

- `cmake --build build/paritychecker_zig --target paritychecker paritychecker_tests -j4`
- `./build/paritychecker_zig/paritychecker_tests`
- `./build/paritychecker_zig/paritychecker --generation --model /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1`
- `./build/paritychecker_zig/paritychecker --generation --model /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1000`

## Deviations from Plan

- None in scope. The maintained proof surface hardened without changing any SML transition table,
  actor ownership, or public C API boundary.

---
*Phase: 24-quantized-path-proof-and-regression*
*Completed: 2026-03-25*
