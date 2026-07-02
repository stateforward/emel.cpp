---
phase: 229-staged-copy-progress-and-completion-semantics
verified: 2026-05-07T22:14:00.000Z
status: verified
requirements_touched:
  - STG-04
  - STG-05
  - STG-06
---

# Phase 229: staged copy, progress, and completion semantics — Verification

## Observable truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Full-span copy writes all `logical_byte_length` bytes deterministically to target (STG-04/STG-06) | verified | Doctest **`io staged_window copies full logical span through fixed chunk tiling (STG-04/06)`** — `memcmp` equality + `bytes_committed == 48`; `done_count == 1`, `error_count == 0` |
| 2 | Non-divisible remainder chunk completes monotone coverage without backtrack (STG-05) | verified | Doctest **`io staged_window copies logical span with non-divisible remainder (STG-04/STG-05)`** — `logical=48`, `chunk=17`, `memcmp` equality; `done_count == 1`, `error_count == 0` |
| 3 | Exactly one terminal success callback fires per accepted dispatch | verified | `done_count == 1` / `error_count == 0` assertions in both copy success cases above |
| 4 | `target_window_bytes >= logical_byte_length` enforced; undersized window rejected before copy | verified | Doctest **`io staged_window rejects logical span larger than declared target window (no overflow)`** — `target_window_bytes=32 < logical_byte_length=48` → **`invalid_target_window`**; canary bytes past window unchanged |
| 5 | Mismatched `source_span_bytes != logical_byte_length` rejected (STG-04 source contract) | verified | Doctest **`io staged_window rejects mismatched caller source_span_bytes (staging)`** → **`source_span_size_mismatch`** (`errors.hpp`; see `lifecycle_tests.cpp` assertion) |
| 6 | Null `source_span` rejected after platform guard (STG-04 source contract) | verified | Doctest **`io staged_window rejects absent source_span after platform guard`** → **`null_source_span`** (`errors.hpp`; see `lifecycle_tests.cpp` assertion) |
| 7 | Aligned/remainder copy path selection via guards only; no branching in actions | verified | `guard_stg_logical_chunk_aligned` / `guard_stg_logical_chunk_remainder` in `guards.hpp`; `sm.hpp` routes to `effect_publish_staged_window_done_aligned` / `_remainder` |
| 8 | No dispatch-local context; actions bounded/allocation-free | verified | `context.hpp` empty; copy actions use only stack locals + `std::memcpy` |

## Automated regression

- `ninja -C build emel_tests_bin` — success (2026-05-07)
- `ctest --test-dir build --output-on-failure -R emel_tests_io` — 100% pass

## Scoped quality gate

- `EMEL_QUALITY_GATES_CHANGED_FILES` scoped to all `src/emel/io/staged_read/*.hpp` + `tests/io/staged_read/lifecycle_tests.cpp`
- Exit: **0** (all lanes pass)
- `lint_snapshot` lane: **passed with no baseline update** — `tests/io/staged_read/lifecycle_tests.cpp` passes
  clang-format and is absent from the lint-failure baseline; `snapshots/lint/clang_format.txt` unchanged from HEAD.
- Coverage: 95.0% lines / 100.0% branches (passing; thresholds: line ≥ 90%, branch ≥ 50%)
- Timing snapshot restored to HEAD after gate run.

## Result

Plan **229-01** satisfies ROADMAP Phase 229 success criteria and **STG-04**, **STG-05**, **STG-06** at the guard/SML
layer with dispatch-through-machine doctest evidence. See **`229-01-SUMMARY.md`** for the canonical changed-file
roster and command transcript.
