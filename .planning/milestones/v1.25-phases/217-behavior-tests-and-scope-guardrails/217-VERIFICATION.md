---
phase: 217-behavior-tests-and-scope-guardrails
status: passed
verified: 2026-05-05T18:50:06Z
requirements:
  - VAL-01
  - VAL-02
---

# Phase 217 Verification

## Requirement Status

| Requirement | Status | Source-Backed Evidence |
|-------------|--------|------------------------|
| VAL-01 | Passed | `io/read`, `io/loader`, `model/tensor`, and `model/loader` doctests drive read/copy success, callback-absent completion/error publication, and representative failure behavior through public `process_event(...)` dispatch and ready-state inspection. |
| VAL-02 | Passed | Source guardrails fail on old staged-policy naming, direct `io/read` event plumbing in `model/loader` or maintained tools, tensor residency ownership moving out of `model/tensor`, and forbidden read-scope widening markers. |

## Verification Commands

- `cmake --build build/zig --target emel_tests_bin` passed.
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_io` passed.
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch`
  passed.
- `build/zig/emel_tests_bin --no-breaks '--source-file=*tests/io/loader/lifecycle_tests.cpp'`
  passed.
- `build/zig/emel_tests_bin --no-breaks '--source-file=*tests/model/tensor/lifecycle_tests.cpp'`
  passed.
- `ctest --test-dir build/zig --output-on-failure -R lint_snapshot` passed after
  `scripts/lint_snapshot.sh --update`.
- `cmake --build build/bench_tools_ninja_generation --target bench_runner_suite_generation generation_compare_tests`
  passed.
- `cmake --build build/bench_tools_ninja_diarization_sortformer --target bench_runner_suite_diarization_sortformer diarization_compare_tests`
  passed.
- `cmake --build build/paritychecker_zig --target paritychecker paritychecker_tests`
  passed.
- `cmake --build build/embedded_size/emel_probe_build --parallel --target emel_qwen3_e2e_probe`
  passed.
- `ctest --test-dir build/bench_tools_ninja_generation --output-on-failure -R generation_compare_tests`
  passed.
- `ctest --test-dir build/bench_tools_ninja_diarization_sortformer --output-on-failure -R diarization_compare_tests`
  passed.
- `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests`
  passed.
- `scripts/check_domain_boundaries.sh` passed.
- `scripts/generate_docs.sh` passed and regenerated `io_loader` architecture output.
- `node .codex/get-shit-done/bin/gsd-tools.cjs validate consistency` passed with the
  pre-existing Phase 211 warning.
- `EMEL_QUALITY_GATES_BENCH_SUITE="generation,diarization_sortformer" EMEL_QUALITY_GATES_CHANGED_FILES="..." scripts/quality_gates.sh`
  passed with scoped `io/loader` coverage at 100.0% line / 87.5% branch,
  generation and diarization Sortformer benchmark snapshots checked, parity/fuzz
  skipped as unrelated by changed-file scope, and docs generation clean.
