---
phase: 228-span-target-window-platform-gating
plan: "01"
subsystem: io-staged-read-validation
tags: [stateforward-sml, staged-read, stg-02, stg-03, plat-02]
requirements-completed: [STG-02, STG-03, PLAT-02]
completed: 2026-05-07
---

# Phase 228 Plan 01: span, target-window, platform gating — Summary

Shipped **guard-only precondition validation** on `emel::io::staged_read::sm` for **`event::staged_window`**:
source span/chunk coherence (STG-02), caller target window sizing against **`stage_chunk_bytes`** (STG-03), and compile-time
platform gate (PLAT-02). No file I/O, mmap, coroutine, copy loop, or tensor residency commits.

## Changed files

- `src/emel/io/staged_read/actions.hpp` — begin/mark/publish/record effects; no runtime branching.
- `src/emel/io/staged_read/context.hpp` — empty machine context (actor-owned state only).
- `src/emel/io/staged_read/detail.hpp` — `staged_window_runtime` + `staged_window_attempt_status` INTERNAL carrier.
- `src/emel/io/staged_read/errors.hpp` — `EMEL_IO_STAGED_READ_PLATFORM_SUPPORTED`, validation error enum entries.
- `src/emel/io/staged_read/events.hpp` — `event::staged_window` (+ `staged_window_request`) and `_done` / `_error`
  outcomes.
- `src/emel/io/staged_read/guards.hpp` — complementary guard predicates; platform uses `if constexpr` on compile macro.
- `src/emel/io/staged_read/sm.hpp` — destination-first validation chain + error publication + unexpected egress rows.
- `tests/io/staged_read/lifecycle_tests.cpp` — precondition + unhappy-path checks exercised **through** **`process_event`** on **`IoStagedRead`** / **`staged_read::sm`** (`dispatch-through-machine` only).

`action::context` remains **empty**. Behavior selection stays in **guards + transition table**.

## Focused verification (driver session)

```text
$ ninja -C build emel_tests_bin
$ ctest --test-dir build --output-on-failure -R emel_tests_io
100% tests passed, 0 tests failed out of 1
Total Test time (real) ~7.7 sec
```

## Notes

- **PLAT-02** unsupported terminal is exercised when **`EMEL_IO_STAGED_READ_PLATFORM_SUPPORTED`** is `0` at compile
  time; supported macOS/Linux/Windows hosts take the supported guard path (guard struct still models PLAT-02 explicitly).

## Validation status

- **228-01 artifacts:** this SUMMARY plus **`228-VERIFICATION.md`** record STG-02/STG-03/PLAT-02 intent and doctest evidence links.
- **Recorded test evidence:** the session transcript above (**`emel_tests_io`** shard). **`scripts/quality_gates.sh` is not claimed or evidenced in this file.**
- **Planner handoff:** next milestone work (**Phase 229** copy/progress semantics) remains on ROADMAP; executor runs repo gates under project policy independent of this summary.
