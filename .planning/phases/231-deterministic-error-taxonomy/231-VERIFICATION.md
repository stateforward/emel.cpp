---
phase: 231-deterministic-error-taxonomy
verified: 2026-05-07T23:00:00.000Z
status: verified
requirements_touched:
  - ESG-01
  - ESG-02A
  - ESG-03
  - ESG-04
requirements_deferred:
  - ESG-02B
---

# Phase 231: deterministic error taxonomy — Verification

## Observable truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Pre-I/O guard failures map to named deterministic categories | verified | `tests/io/staged_read/lifecycle_tests.cpp` asserts `invalid_callbacks`, `invalid_stage_contract`, and `invalid_target_window` through `process_event(...)` callbacks; `unsupported_platform` is present as a named category plus explicit guard/action/transition in source/generated architecture, but not forced by supported-host doctests in this phase |
| 2 | Source-contract read-surface failures map to deterministic categories (ESG-02A) | verified | tests assert `null_source_span`, `source_span_size_mismatch`, `insufficient_source_span` through `process_event(...)` |
| 3 | Stage-contract/sequence failure taxonomy is explicit and deterministic | verified | `invalid_stage_contract` assertions for zero-length/chunk overflow/offset-overflow paths |
| 4 | Staged-read boundary remains exception-free (ESG-04) | verified | `event::staged_window` constructor is `noexcept`; callbacks/tests execute without throw pathways; no exceptions introduced in staged_read code |
| 5 | ESG-02B file open/seek/read taxonomy is deferred transparently | verified | `.planning/REQUIREMENTS.md` + `.planning/ROADMAP.md` record ESG-02B as Deferred/Future with file-backed staged source precondition |
| 6 | Generated architecture docs reflect the Phase 231 source-span taxonomy graph | verified | `.planning/architecture/io_staged_read.md` and `.planning/architecture/mermaid/io_staged_read.mmd` include explicit null/mismatch/insufficient-source error branches |

## Required command evidence

```text
$ ninja -C build emel_tests_bin
exit 0
```

```text
$ ctest --test-dir build --output-on-failure -R emel_tests_io
100% tests passed, 0 tests failed out of 1
```

```text
$ LC_ALL=C EMEL_QUALITY_GATES_CHANGED_FILES='src/emel/io/staged_read/errors.hpp:src/emel/io/staged_read/guards.hpp:src/emel/io/staged_read/actions.hpp:src/emel/io/staged_read/sm.hpp:tests/io/staged_read/lifecycle_tests.cpp' scripts/quality_gates.sh
exit 0
coverage lines: 95.7% (132/138)
coverage branches: 86.4% (19/22)
```

## Snapshot handling

- `snapshots/quality_gates/timing.txt` was modified by quality gate execution and restored to `HEAD`.
- `snapshots/lint/clang_format.txt` remained unchanged.
- No snapshot baseline update was performed.
