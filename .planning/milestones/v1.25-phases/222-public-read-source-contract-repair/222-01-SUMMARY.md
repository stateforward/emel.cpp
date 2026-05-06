---
phase: 222-public-read-source-contract-repair
plan: 01
status: complete
completed: 2026-05-06T04:46:52Z
requirements:
  - PLAT-01
  - TIO-03
  - VAL-02
  - VAL-04
---

# Phase 222 Summary

## Completed

Phase 222 removed maintained-lane actor-detail reach-through by adding
`emel::io::source::load_file_bytes` in `src/emel/io/source/any.hpp` and removing
the setup-time file loader from `src/emel/io/read/detail.hpp`.

## Source Changes

- Added public setup-time source-byte loading under `emel::io::source`.
- Rewired maintained generation, Sortformer diarization, embedded probe, and
  paritychecker lanes to call `emel::io::source::load_file_bytes`.
- Updated model-loader guardrails to require the public source API and forbid
  `emel/io/read/detail.hpp` in maintained tool lanes.
- Updated paritychecker guardrails to prove `parity_engines.cpp` uses the public
  source API and does not include `emel/io/read/detail.hpp`.

## Evidence

- `cmake --build build/zig --target emel_tests_bin` passed.
- `cmake --build build/paritychecker_zig --target paritychecker_tests` passed.
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch`
  passed.
- `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests`
  passed.
- `ctest --test-dir build/bench_tools_ninja_generation --output-on-failure -R generation_compare_tests`
  passed after refreshing the reference build cache to use `/usr/bin/git`
  instead of the atmux Git shim.
- `scripts/check_domain_boundaries.sh` passed.
- Changed-file scoped `scripts/quality_gates.sh` exited 0 with no
  benchmark-regression override.
