---
phase: 34-initializer-surface-shrink-and-proof
plan: 03
subsystem: emel-e2e-probe
tags: [embedded-size, qwen3, probe, tokenizer, binary-size]
requires:
  - phase: 33-fixture-metadata-and-contract-lock
    provides: locked maintained workload and comparator boundary
provides:
  - EMEL-owned Qwen3 E2E probe executable
  - removal of fallback vocab sentinel binary bloat
  - corrected executable-size harness inputs for EMEL row
affects: [35-01 comparator proof, 36-01 publication plumbing, 39-01 snapshot refresh]
tech-stack:
  added: [src/emel/text/unicode_data.cpp]
  patterns: [native GGUF vocab path, executable-size harness, out-of-line constant data]
key-files:
  created: [src/emel/text/unicode_data.cpp]
  modified:
    [scripts/embedded_size.sh, tools/embedded_size/emel_probe/main.cpp, src/emel/text/encoders/events.hpp, src/emel/text/encoders/guards.hpp, src/emel/text/encoders/bpe/detail.hpp, src/emel/text/unicode_data.hpp, tests/text/encoders/common_tests.cpp]
key-decisions:
  - "The published EMEL path must stay EMEL-owned end to end and must not use llama.cpp bootstrap."
  - "Huge empty-vocab sentinels were removed instead of accepted as part of the executable image."
  - "Unicode table data moved out of headers, but the dominant size fix was eliminating redundant 26 MB fallback vocab objects."
requirements-completed: []
duration: 0min
completed: 2026-04-02
---

# Phase 34 Plan 03 Summary

**The EMEL Qwen3 E2E probe is now a truthful final executable measurement**

## Accomplishments

- Verified the maintained EMEL probe path in
  [main.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/embedded_size/emel_probe/main.cpp)
  runs the Qwen3 `hello` -> first-token slice through EMEL-owned loading and generation code.
- Removed the redundant fallback vocab sentinels in
  [events.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/text/encoders/events.hpp),
  [guards.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/text/encoders/guards.hpp),
  and
  [detail.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/text/encoders/bpe/detail.hpp),
  which dropped the EMEL probe from `56,637,960` bytes to `4,073,016` bytes in the local
  executable-size run.
- Moved unicode constant data out of headers into
  [unicode_data.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/text/unicode_data.cpp)
  and updated encoder tests to pass explicit vocab references.

## Verification

- `cmake --build build/debug --target emel_tests_bin`
- `./build/debug/emel_tests_bin --test-case='*encoder*'`
- `./build/debug/emel_tests_bin --test-case='*tokenizer*'`
- `./scripts/embedded_size.sh --json`

## Deviations from Plan

- The main executable-size fix was not heap placement or unicode extraction alone; the decisive
  change was eliminating the two 26 MB fallback vocab singletons discovered in the linker map.

---
*Phase: 34-initializer-surface-shrink-and-proof*
*Completed: 2026-04-02*
