---
phase: 194-maintained-loader-tool-prebind
status: passed
verified: 2026-05-03T04:38:04Z
requirements:
  - TENSOR-02
  - LOAD-02
---

# Phase 194 Verification

## Verdict

`passed`

The maintained generation benchmark, Sortformer benchmark, embedded-size probe, and paritychecker
now prebind GGUF KV storage before model-loader dispatch. Source-backed regression coverage verifies
the maintained parse callbacks no longer call `.kv_arena.resize` or `.kv_entries.resize`.

## Must-Have Checks

| Must Have | Status | Evidence |
|-----------|--------|----------|
| Maintained lanes size GGUF KV storage before `model_loader.process_event(...)` | PASS | `prebind_emel_gguf_storage(...)` in generation, Sortformer, and embedded probe; `prebind_gguf_kv_storage(...)` in paritychecker. |
| Parse callbacks no longer grow KV storage | PASS | `tests/model/loader/lifecycle_tests.cpp` extracts callback bodies and rejects `.kv_arena.resize` / `.kv_entries.resize`. |
| Maintained benchmark/parity/probe validation passes | PASS | Focused ctest commands and corrected scoped quality gate passed. |
| Approved snapshot updates are truthful | PASS | `snapshots/bench/benchmarks.txt` updated via `scripts/bench.sh --snapshot --update --suite=generation`; follow-up quality gate passed without `EMEL_QUALITY_GATES_ALLOW_BENCH_REGRESSION`. |

## Commands

- `cmake --build build/zig --target emel_tests_bin`
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch`
- `cmake --build build/paritychecker_zig --target paritychecker_tests`
- `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests`
- `scripts/bench.sh --snapshot --update --suite=generation`
- `EMEL_QUALITY_GATES_CHANGED_FILES='tools/bench/generation_bench.cpp:tools/bench/diarization/sortformer_fixture.hpp:tools/embedded_size/emel_probe/main.cpp:tools/paritychecker/parity_engines.cpp:tests/model/loader/lifecycle_tests.cpp:snapshots/bench/benchmarks.txt' EMEL_QUALITY_GATES_BENCH_SUITE='generation,diarization_sortformer' scripts/quality_gates.sh`

## Notes

The corrected quality-gate run selected:

- benchmark suites `generation` and `diarization_sortformer`
- full paritychecker runner because `tools/paritychecker/parity_engines.cpp` changed
- docs generation because the benchmark snapshot changed

The gate passed with benchmark regressions enforced normally.
