---
phase: 219-maintained-read-source-provenance
plan: 01
status: complete
completed: 2026-05-05T21:15:00Z
requirements:
  - PLAT-01
  - TIO-03
  - VAL-04
---

# Phase 219 Summary

## Completed

Phase 219 replaced maintained tool-local full-file read scaffolds with a
maintained source-byte contract and added guardrails that
fail if the maintained generation, Sortformer diarization, embedded probe, or
paritychecker lanes reintroduce `read_file_bytes` helpers for read/copy
evidence.

## Implementation

- Added a maintained setup-time source-byte loading helper with `io/read` error
  categories. Phase 222 later moved this helper from actor-internal
  `src/emel/io/read/detail.hpp` to public `src/emel/io/source/any.hpp` as
  `emel::io::source::load_file_bytes`.
- Rewired maintained EMEL lanes in:
  - `tools/bench/generation_bench.cpp`
  - `tools/bench/diarization/sortformer_fixture.hpp`
  - `tools/embedded_size/emel_probe/main.cpp`
  - `tools/paritychecker/parity_engines.cpp`
- Removed the paritychecker common `read_file_bytes` API from:
  - `tools/paritychecker/parity_assets.hpp`
  - `tools/paritychecker/parity_assets.cpp`
- Updated guardrails and source-backed checks in:
  - `tests/model/loader/lifecycle_tests.cpp`
  - `tools/paritychecker/paritychecker_tests.cpp`

## Workflow Note

The first source edits for this phase were made before the phase context and
plan existed. That ordering mistake was reconciled in `219-CONTEXT.md` and
`219-01-PLAN.md`; the implementation above is now captured by this phase's
planning artifacts.
