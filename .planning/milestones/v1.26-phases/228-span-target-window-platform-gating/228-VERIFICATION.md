---
phase: 228-span-target-window-platform-gating
verified: 2026-05-07T21:25:00.000Z
status: verified
requirements_touched:
  - STG-02
  - STG-03
  - PLAT-02
---

# Phase 228: span, target-window, platform gating — Verification

## Observable truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | **Oversized `stage_chunk_bytes`:** rejects when **`stage_chunk_bytes > logical_byte_length`** (STG-02) before pre-ready | verified | Doctest **`io staged_window rejects stage chunk larger than logical span (STG-02)`** (`logical_byte_length = 24`, `stage_chunk_bytes = 64` → **`invalid_staging_contract`**); aligns with **`guard_stg_source_contract_*`** (`guards.hpp`, `stage_chunk_bytes > logical_byte_length`) |
| 2 | Invalid source span baseline (non-zero lengths, overflow) | verified | Doctests **`io staged_window rejects zero logical span (STG-02)`**, **`io staged_window rejects zero stage chunk bytes (STG-02)`**, **`io staged_window rejects uint64 span overflow at file_offset + logical length`**; errors via **`invalid_staging_contract`** where applicable |
| 3 | **`target_window_bytes >= stage_chunk_bytes`**: accepted when predicate holds | verified | Doctest **`io staged_window validation accepts coherent span,target,platform guards`** — `stage_chunk_bytes = 16`, `target_window_bytes = 128` (buffer size), **`process_event` succeeds** and **`staged_window_done`** path runs |
| 4 | **`target_window_bytes >= stage_chunk_bytes`**: rejected when predicate fails | verified | Doctest **`io staged_window rejects target window smaller than stage chunk (STG-03)`** — `stage_chunk_bytes = 16`, `target_window_bytes = 8`, error **`invalid_target_window`** |
| 5 | Invalid target buffer (null) rejected | verified | Doctest **`io staged_window rejects null target buffer (STG-03)`** |
| 6 | Unsupported platform fail-closed via guarded transition | verified | **`guard_platform_staged_read_*`** + **`EMEL_IO_STAGED_READ_PLATFORM_SUPPORTED`** in **`errors.hpp`**; supported hosts take the supported transition row |
| 7 | No dispatch-local context; validation-only orchestration slice | verified | **`context.hpp`** empty; **`staged_read`** sources contain no read/mmap/syscall surface |

## Automated regression

- `ninja -C build emel_tests_bin` — success (driver session, 2026-05-07)
- `ctest --test-dir build --output-on-failure -R emel_tests_io` — pass (same session)

This verification ledger **does not** record a **`scripts/quality_gates.sh`** run or assert organization-wide CI gate passage.

## Result

Plan **228-01** meets the ROADMAP Phase 228 precondition goals for **STG-02**, **STG-03**, and **PLAT-02** at the guard/SML layer, with doctest evidence summarized above. **`228-01-SUMMARY.md`** holds the canonical changed-file roster and **`emel_tests_io`** command transcript.
