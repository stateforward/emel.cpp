---
phase: 248
plan: 01
status: complete
requirements-completed:
  - PERF-01
---

# Phase 248 Summary

## Completed

- Added a maintained `io/loader` cooperative async route backed by an injected
  `emel::io::async::sm`.
- Treated `cooperative_async` as storage-backed in `model/loader` so tensor target storage remains
  request-owned before tensor apply commits residency.
- Wired the generation benchmark fixture through the public loader strategy boundary using
  `emel::IoAsync` and public `io/loader` contracts.
- Added focused `io/loader` and `model/loader` tests for cooperative async single/batch dispatch and
  maintained end-to-end success.
- Published source-backed performance evidence:
  `EMEL_MODEL_LOAD_IO_STRATEGY=cooperative_async scripts/bench.sh --snapshot --compare --suite=generation`
  completed with `488169958 ns/op` EMEL vs `353585500 ns/op` reference (`1.381x`).

## Constrained-RAM Value

For large models in constrained RAM, this path matters because the maintained loader now has a
bounded-window async strategy instead of an unsupported placeholder. Tensor planning assigns
caller-owned destination storage, and the public async actor advances chunks into that storage before
`model/tensor` commits residency. That does not make this benchmark faster on the recorded fixture,
but it proves the maintained route that avoids treating "unsupported" as performance evidence and
preserves the architecture needed to keep model loading bounded instead of requiring an async
result to be a fully staged payload.

## Verification

- `cmake --build build/debug --target emel_tests_bin` — passed.
- `./build/debug/emel_tests_bin --test-case="*io loader*" --test-case="*model loader*"` — passed,
  34 test cases / 683 assertions.
- `EMEL_MODEL_LOAD_IO_STRATEGY=cooperative_async scripts/bench.sh --snapshot --compare --suite=generation`
  — passed, recorded measured `cooperative_async` timing.
- `ctest --test-dir build/debug -R "emel_tests|lint_snapshot" --output-on-failure` — passed after
  formatting, 14/14 targets.
- `EMEL_QUALITY_GATES_BENCH_SUITE="generation" EMEL_QUALITY_GATES_CHANGED_FILES="..." scripts/quality_gates.sh`
  — passed without benchmark-regression override; generation benchmark lane recorded
  `425780125 ns/op` EMEL vs `316074125 ns/op` reference (`1.347x`) during the scoped gate.
