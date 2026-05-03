---
phase: 194-maintained-loader-tool-prebind
plan: 01
status: complete
completed: 2026-05-03T04:38:04Z
requirements-completed:
  - TENSOR-02
  - LOAD-02
---

# Phase 194 Summary

## Result

Closed the maintained-path loader dispatch allocation gap found by the v1.22 audit.

The generation benchmark, Sortformer benchmark, embedded-size probe, and paritychecker now probe
GGUF requirements and resize KV arena / entry storage before `model_loader.process_event(...)`.
Their loader parse callbacks now only bind the already-sized storage, parse, copy tensor names, and
populate metadata.

## Code Changes

- `tools/bench/generation_bench.cpp`: added fixture-owned `gguf_tensor_count` and
  `prebind_emel_gguf_storage(...)`; `run_emel_parse_model(...)` no longer grows KV storage.
- `tools/bench/diarization/sortformer_fixture.hpp`: added the same prebind path for the maintained
  Sortformer benchmark fixture.
- `tools/embedded_size/emel_probe/main.cpp`: added pre-dispatch GGUF prebind for the embedded-size
  maintained probe.
- `tools/paritychecker/parity_engines.cpp`: split `prebind_gguf_kv_storage(...)` from
  `parse_gguf_kv_storage(...)` and call prebind before both generation load and vocab-only GGUF
  parsing.
- `tests/model/loader/lifecycle_tests.cpp`: added a source-backed regression test that extracts the
  maintained parse callback bodies and rejects `.kv_arena.resize` / `.kv_entries.resize`.
- `snapshots/bench/benchmarks.txt`: updated the generation benchmark snapshot through
  `scripts/bench.sh --snapshot --update --suite=generation` after the corrected quality gate found
  a generation snapshot regression.

## Validation

- `cmake --build build/zig --target emel_tests_bin` passed.
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch` passed.
- `cmake --build build/paritychecker_zig --target paritychecker_tests` passed.
- `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests` passed.
- `scripts/bench.sh --snapshot --update --suite=generation` updated the approved generation
  snapshot baseline.
- Corrected scoped quality gate passed:
  `EMEL_QUALITY_GATES_CHANGED_FILES='tools/bench/generation_bench.cpp:tools/bench/diarization/sortformer_fixture.hpp:tools/embedded_size/emel_probe/main.cpp:tools/paritychecker/parity_engines.cpp:tests/model/loader/lifecycle_tests.cpp:snapshots/bench/benchmarks.txt' EMEL_QUALITY_GATES_BENCH_SUITE='generation,diarization_sortformer' scripts/quality_gates.sh`.

## Requirement Closure

- `TENSOR-02`: satisfied for the maintained path because tensor-owned residency orchestration no
  longer depends on dispatch-time GGUF KV storage allocation in the maintained benchmark, probe, or
  parity tool callbacks.
- `LOAD-02`: satisfied for the maintained path because the loader dispatch coordinates already
  prepared tensor-owned storage and does not force tool callbacks to grow GGUF KV storage inside
  the actor RTC chain.
