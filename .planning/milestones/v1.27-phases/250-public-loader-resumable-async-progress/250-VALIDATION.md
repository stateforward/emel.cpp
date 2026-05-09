---
phase: 250
status: complete
validated: 2026-05-09
---

# Phase 250 Validation

## Result

Validated by focused doctests, cooperative async benchmark/compare, parity, docs generation, and
changed-file scoped quality gates.

## Commands

- `cmake --build build/debug --target emel_tests_bin` — passed.
- `./build/debug/emel_tests_bin --test-case="*io loader*"` — passed.
- `./build/debug/emel_tests_bin --test-case="*model loader*cooperative async*"` — passed.
- `EMEL_QUALITY_GATES_BENCH_SUITE="generation" EMEL_QUALITY_GATES_CHANGED_FILES="src/emel/io/loader/events.hpp:src/emel/io/loader/detail.hpp:src/emel/io/loader/actions.hpp:src/emel/io/loader/guards.hpp:src/emel/io/loader/sm.hpp:src/emel/io/loader/context.hpp:src/emel/model/loader/events.hpp:src/emel/model/loader/actions.hpp:src/emel/model/loader/guards.hpp:src/emel/model/loader/sm.hpp:tests/io/loader/lifecycle_tests.cpp:tests/model/loader/lifecycle_tests.cpp:tools/bench/generation_bench.cpp:tools/bench/model_load_strategy.hpp:snapshots/lint/clang_format.txt" scripts/quality_gates.sh` — passed.
- `EMEL_MODEL_LOAD_IO_STRATEGY=cooperative_async scripts/bench.sh --snapshot --compare --suite=generation` — passed.
