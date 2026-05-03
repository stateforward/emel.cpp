# Phase 194 Validation: Maintained Loader Tool Prebind

**Status:** Passed
**Validated:** 2026-05-03

## Scope

Phase 194 closed the maintained-path allocation gap for generation benchmark,
Sortformer benchmark, embedded-size probe, and paritychecker model-loading lanes.
The validation evidence is source-backed, not artifact-only.

## Source Evidence

- `tools/bench/generation_bench.cpp` prebinds GGUF KV storage before
  `model_loader.process_event(...)`.
- `tools/bench/diarization/sortformer_fixture.hpp` prebinds GGUF KV storage before
  `model_loader.process_event(...)`.
- `tools/embedded_size/emel_probe/main.cpp` prebinds GGUF KV storage before
  `model_loader.process_event(...)`.
- `tools/paritychecker/parity_engines.cpp` prebinds GGUF KV storage before
  `model_loader.process_event(...)`.
- `tests/model/loader/lifecycle_tests.cpp` includes source-backed regression coverage rejecting
  `.kv_arena.resize` and `.kv_entries.resize` inside the maintained parse callback bodies.

## Validation Commands

- `scripts/check_domain_boundaries.sh` passed.
- `cmake --build build/zig --target emel_tests_bin` passed.
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch` passed.
- `cmake --build build/paritychecker_zig --target paritychecker_tests` passed.
- `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests` passed.

## Result

The missing Phase 194 validation artifact gap is closed. Phase 195 carries the stricter audit
closeout for the loader/tensor outcome contract and tensor wrapper rule conflicts found after this
phase.
