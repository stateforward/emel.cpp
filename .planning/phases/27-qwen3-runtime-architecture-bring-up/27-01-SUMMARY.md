---
phase: 27-qwen3-runtime-architecture-bring-up
plan: 01
subsystem: model
tags: [qwen3, gguf, execution-view, quantized-audit]
requires:
  - phase: 26
    provides: canonical qwen3 fixture anchoring and structured formatter/request contract
provides:
  - explicit canonical qwen3 execution-view binding with required q/k norm tensors
  - explicit qwen3 quantized-path audit stages for attention_q_norm and attention_k_norm
  - architecture-guarded not_applicable audit reporting outside qwen3
affects: [27-02, 28, 29]
tech-stack:
  added: []
  patterns:
    - architecture-specific execution-view binding at the model/data boundary
    - qwen3-only audit stages reported as not_applicable outside qwen3
key-files:
  created:
    - .planning/phases/27-qwen3-runtime-architecture-bring-up/27-01-SUMMARY.md
  modified:
    - tests/model/loader/lifecycle_tests.cpp
    - tests/generator/lifecycle_tests.cpp
    - src/emel/model/llama/detail.hpp
    - src/emel/model/data.cpp
key-decisions:
  - "Treat qwen3 as an explicit execution architecture instead of a renamed llama alias."
  - "Require blk.%d.attn_q_norm.weight and blk.%d.attn_k_norm.weight for canonical qwen3 execution-view construction."
  - "Publish qwen3-only audit stages as not_applicable outside qwen3 to preserve llama audit truth."
patterns-established:
  - "Model architecture branching belongs in src/emel/model/data.cpp, not in generator or tool heuristics."
  - "Quantized audit expansions must preserve prior architecture claims explicitly rather than broadening shared counts."
requirements-completed: [RUN-02]
duration: 25 min
completed: 2026-03-28
---

# Phase 27 Plan 01: Canonical Qwen3 Execution View Summary

**Canonical `qwen3` execution-view binding now requires explicit Q/K norm tensors and publishes truthful Qwen3-only audit stages without regressing the maintained Llama slice.**

## Performance

- **Duration:** 25 min
- **Started:** 2026-03-28T00:14:23-05:00
- **Completed:** 2026-03-28T00:40:05-05:00
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Added fail-first doctest coverage for canonical `qwen3` execution-view construction and Q/K norm audit stages.
- Extended the maintained execution-view contract to bind `attn_q_norm.weight` and `attn_k_norm.weight` only for the canonical `qwen3` block contract.
- Expanded the quantized-path audit inventory to include `attention_q_norm` and `attention_k_norm`, while reporting those stages as `not_applicable` for non-`qwen3` architectures.

## Task Commits

Each task was committed atomically:

1. **Task 1: Reproduce missing Qwen3 execution-view support in tests** - `80424f4` (test)
2. **Task 2: Add explicit Qwen3 execution-view and audit support** - `a63a5bf` (feat)

**Plan metadata:** not committed (`commit_docs=false`)

## Files Created/Modified

- `tests/model/loader/lifecycle_tests.cpp` - adds canonical `qwen3` execution-view success/failure coverage around required Q/K norm tensors
- `tests/generator/lifecycle_tests.cpp` - adds Qwen3 quantized-path audit coverage for explicit Q/K norm stages
- `src/emel/model/llama/detail.hpp` - extends block-view and quantized-stage enums for Qwen3 Q/K norm tensors
- `src/emel/model/data.cpp` - adds explicit `qwen3` architecture support, required tensor binding, stage naming, and non-`qwen3` audit handling

## Decisions Made

- `qwen3` is handled as an explicit execution architecture branch in the maintained execution-view helpers.
- Canonical Qwen3 execution-view construction fails with `model_invalid` if either required Q/K norm tensor is missing.
- Qwen3-only audit stages stay visible in the shared inventory but are marked `not_applicable` outside `qwen3` to preserve prior Llama audit truth.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Preserved non-Qwen audit truth after adding Qwen3-only stages**
- **Found during:** Task 2 (Add explicit Qwen3 execution-view and audit support)
- **Issue:** The first Task 2 pass widened the shared audit inventory without making non-`qwen3` handling explicit, which regressed maintained Llama audit expectations under `scripts/quality_gates.sh`.
- **Fix:** Marked `attention_q_norm` and `attention_k_norm` as `not_applicable` outside the canonical `qwen3` block contract instead of treating them as missing shared stages.
- **Files modified:** `src/emel/model/data.cpp`
- **Verification:** `./build/zig/emel_tests_bin --test-case='*qwen3*execution*,*qwen3*quantized*' --no-breaks`, `./build/zig/emel_tests_bin --test-case='*generator_quantized_path_audit*' --no-breaks`
- **Committed in:** `a63a5bf` (part of Task 2 commit)

---

**Total deviations:** 1 auto-fixed (maintained-audit truth preservation)
**Impact on plan:** Necessary correctness fix. Scope stayed inside the planned execution-view/audit boundary.

## Issues Encountered

The full repo gate still fails in `tools/paritychecker/paritychecker_tests.cpp` because maintained parity and generation surfaces have not been brought up for canonical Qwen3 yet. That is expected Phase `27-02` work, not a regression in the `27-01` execution-view slice.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Wave 2 can now consume an explicit canonical Qwen3 execution view with required Q/K norm tensors and truthful audit reporting. The remaining repo-gate failures are concentrated on the maintained generator/parity path, which is the intended scope of `27-02`.

---
*Phase: 27-qwen3-runtime-architecture-bring-up*
*Completed: 2026-03-28*
