---
phase: 229-staged-copy-progress-and-completion-semantics
plan: "01"
subsystem: io-staged-read-copy
tags: [stateforward-sml, staged-read, stg-04, stg-05, stg-06]
requirements-completed: [STG-04, STG-05, STG-06]
completed: 2026-05-07
---

# Phase 229 Plan 01: staged copy, progress, and completion semantics — Summary

Shipped **guard-routed deterministic chunked copy** on `emel::io::staged_read::sm` for `event::staged_window`:
full-span tiling with aligned chunk stride (STG-04/STG-06) and remainder-chunk handling for non-divisible spans
(STG-05). Exactly one terminal success callback per accepted dispatch with `bytes_committed == logical_byte_length`.

## Blocker resolved

`guard_stg_target_window_valid` now enforces `target_window_bytes >= logical_byte_length` (full-span commitment
requires caller to own the full destination window, not just the stage chunk slab). Aligned and remainder copy
paths are guard-routed via `guard_stg_logical_chunk_aligned` / `guard_stg_logical_chunk_remainder` in `sm.hpp`;
copy execution lives in bounded, allocation-free `effect_publish_staged_window_done_aligned` /
`effect_publish_staged_window_done_remainder` actions.

## Changed files

- `src/emel/io/staged_read/actions.hpp` — `effect_publish_staged_window_done_aligned` + `_remainder`: deterministic
  monotone chunk tiling; single `on_done` invocation; `bytes_committed = logical_byte_length`.
- `src/emel/io/staged_read/guards.hpp` — `guard_stg_copy_span_*` (source span validity), `guard_stg_logical_chunk_aligned`
  / `_remainder` (copy path routing).
- `src/emel/io/staged_read/sm.hpp` — `state_guard_copy_source_decision` state + transitions routing to aligned/remainder
  copy effects; `state_staged_pre_ready` stays as guard acceptance placeholder before copy dispatch.
- `src/emel/io/staged_read/events.hpp` — `staged_window_request` extended with `source_span` + `source_span_bytes`.
- `tests/io/staged_read/lifecycle_tests.cpp` — full-span tiling copy test (STG-04/06), non-divisible remainder test
  (STG-04/05), `target_window_bytes >= logical_byte_length` enforcement test, mismatched/null source span rejection
  tests; `done_count`/`error_count` counters asserting exactly one terminal success per dispatch.

`action::context` remains **empty**. Copy path selection stays in **guards + transition table** only.

## Focused verification

```text
$ ninja -C build emel_tests_bin
ninja: no work to do.
$ ctest --test-dir build --output-on-failure -R emel_tests_io
100% tests passed, 0 tests failed out of 1
$ EMEL_QUALITY_GATES_CHANGED_FILES='<staged_read sources + test>' scripts/quality_gates.sh
exit 0  (coverage 95.0% lines / 100.0% branches; lint_snapshot clean)
```

## Notes

- `tests/io/staged_read/lifecycle_tests.cpp` passes **clang-format** — no lint snapshot baseline update was needed.
  The file is absent from the lint-failure baseline (`snapshots/lint/clang_format.txt` unchanged from HEAD).
- POSIX file-descriptor staging (reading actual bytes from disk) remains **Phase 230+** work; `source_span` currently
  requires caller-supplied bytes matching the `io/read` externally-owned posture.
- PLAT-02 unsupported terminal exercised when `EMEL_IO_STAGED_READ_PLATFORM_SUPPORTED` is `0` at compile time;
  supported macOS/Linux/Windows hosts take the supported guard path.

## Next

Phase **230** context lifetime and LIFE/ESG path semantics per ROADMAP.
