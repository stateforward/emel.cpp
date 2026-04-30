---
phase: 117
status: passed
verified: 2026-04-27
requirements:
  - REOPEN-01
---

# Phase 117 Verification

## Requirement Verification

| Requirement | Result | Evidence |
|-------------|--------|----------|
| REOPEN-01 | passed | `tools/bench/whisper_compare.py` now returns nonzero for `bounded_drift`, and focused tests prove transcript mismatch fails. |

## Command Evidence

```sh
cmake --build build/whisper_compare_tools --target whisper_benchmark_tests -j 4
build/whisper_compare_tools/whisper_benchmark_tests
```

Result: 9 test cases and 130 assertions passed.

```sh
python3 -m py_compile tools/bench/whisper_compare.py tools/bench/whisper_benchmark.py
```

Result: passed.

```sh
scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build
```

Result: `status=exact_match reason=ok`.

```sh
EMEL_QUALITY_GATES_CHANGED_FILES='tools/bench/whisper_compare.py:tools/bench/whisper_benchmark_tests.cpp:tools/bench/CMakeLists.txt' \
  EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare scripts/quality_gates.sh
```

Result: passed. The gate skipped unrelated lanes and ran the maintained Whisper compare suite.

```sh
scripts/check_domain_boundaries.sh
```

Result: passed.

## Source Checks

- `bounded_drift` remains a machine-readable compare status.
- Only `exact_match` returns process exit 0.
- No model-family domain placement changed.
