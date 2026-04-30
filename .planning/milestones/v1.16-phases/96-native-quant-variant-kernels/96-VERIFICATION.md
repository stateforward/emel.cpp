---
phase: 96-native-quant-variant-kernels
verified: 2026-04-25T20:05:00Z
status: passed
score: 4/4 must-haves verified
---

# Phase 96: Native Quant Variant Kernels Verification Report

**Phase Goal:** Add native kernel-level support for the narrowed maintained Whisper quant
variant family `{q8_0, q4_0, q4_1}` (per Phase 95 narrowing approved 2026-04-25, agent-0
message `J2bGANuAk61CW6P6`), including ARM optimized dispatch proof.
**Verified:** 2026-04-25T20:05:00Z
**Status:** passed
**Approved scope:** Option A (FULL NEON), agent-0 message `tf9ePyZNIHVRJYWD`.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `q4_0`, `q4_1`, and `q8_0` execute through native EMEL kernel-owned code (the narrowed v1.16 maintained Whisper tiny GGUF variant family). | ✓ VERIFIED | `q8_0` paths already existed in `src/emel/kernel/aarch64/actions.hpp` (NEON vector + multiple packed forms). Phase 96 added real NEON intrinsics for `dot_q4_0_q8_0_row_neon` / `_4rows_neon` and `dot_q4_1_q8_0_row_neon` / `_4rows_neon` plus matching `execute_neon_mul_mat_q4_*_vector` paths. Numeric parity vs scalar within `1.0e-5f` is asserted by the new doctest cases (118/118 assertions). |
| 2 | Runtime dtype selection is modeled through explicit guards/transitions instead of action/detail routing. | ✓ VERIFIED | New guards `simd_op_mul_mat_q4_0_vector` and `simd_op_mul_mat_q4_1_vector` in `src/emel/kernel/aarch64/guards.hpp` decide which path runs based on `dtype_code(src0.type)` plus `validate_dispatch_request`. New transitions in `src/emel/kernel/aarch64/sm.hpp` are destination-first and route via the guard set. The generic fall-through guard explicitly excludes q4_0/q4_1 vector lanes so the generic backend cannot steal dispatch. Action functors only execute the already-chosen kernel; they do not branch on behavior. |
| 3 | Quantized hot paths do not use whole-tensor dequantize-to-f32 fallback. | ✓ VERIFIED | The new `execute_neon_mul_mat_q4_*_vector_unchecked` paths quantize the f32 src1 to per-block `q8_0` on the stack (`alignas(64) std::array<block_q8_0, MAX_Q8_0_BLOCKS>`) and feed the q4_* LHS blocks directly through the NEON dot-product without dequantizing to f32. The dot helpers operate on packed 4-bit values; no whole-tensor f32 buffer is materialized for src0. |
| 4 | ARM dispatch attribution and focused kernel tests prove the maintained paths. | ✓ VERIFIED | New context counters `optimized_q4_0_dispatch_count`, `optimized_q4_0_vector_dispatch_count`, `optimized_q4_1_dispatch_count`, `optimized_q4_1_vector_dispatch_count` are exposed via `aarch64::sm` accessor methods. New doctest cases `kernel_aarch64_q4_0_vector_route_is_explicit_and_numeric_match` and `_q4_1_vector_*` drive the SML facade with `op_mul_mat` and assert the relevant counters increment exactly once and the shared (non-optimized) counter stays zero. The build host has `__ARM_FEATURE_DOTPROD` defined so the DOTPROD-fast `vdotq_s32` path is exercised. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `96-CONTEXT.md` | Source-backed phase context | ✓ EXISTS + SUBSTANTIVE | Includes approved-scope reference, current kernel landscape per dtype, block layouts, what Phase 96 must add, constraints, definition of done. |
| `96-01-PLAN.md` | Tasked execution plan | ✓ EXISTS + SUBSTANTIVE | Nine explicit tasks with acceptance criteria, risk notes, AGENTS.md routing constraints, build host attribution. |
| `96-01-SUMMARY.md` | Execution summary | ✓ EXISTS + SUBSTANTIVE | Documents outcomes per task, includes the discovered `is_native_quantized_dtype` extension, file change table, DOTPROD attribution on host, coverage-scoping limitation, residual risk, snapshot notes. |
| Updated `.planning/ROADMAP.md` | Narrowed Phase 96 SC1 | ✓ EXISTS + SUBSTANTIVE | SC1 explicitly lists `{q4_0, q4_1, q8_0}` and references the agent-0 narrowing approval. |
| Updated `src/emel/kernel/detail.hpp` | `quantize_row_q4_0_ref`, `quantize_row_q4_1_ref`, `is_native_quantized_dtype` extension, `<limits>` include | ✓ EXISTS + SUBSTANTIVE | Quant-helpers mirror q5_0 pattern; predicate now includes q4_0/q4_1 alongside q5_0/q8_0/k-dtypes. |
| Updated `src/emel/kernel/aarch64/actions.hpp` | NEON intrinsics + execute paths + action structs | ✓ EXISTS + SUBSTANTIVE | Real `vdotq_s32` / `vmull_s8` fallback for q4_0 and q4_1; per-row and 4-row variants; `horizontal_sum_s8_neon` helper for q4_1's rhs sum; vector execute paths mirror q5_0 vector. |
| Updated `src/emel/kernel/aarch64/guards.hpp` | New guards + updated generic fall-through | ✓ EXISTS + SUBSTANTIVE | `simd_op_mul_mat_q4_0_vector` and `_q4_1_vector` added; `simd_op_mul_mat_generic` fall-through excludes them. |
| Updated `src/emel/kernel/aarch64/sm.hpp` | Two new transitions + four new accessor methods | ✓ EXISTS + SUBSTANTIVE | Destination-first transitions placed after q5_0_vector. Accessors exposed for the four new counters. |
| Updated `src/emel/kernel/aarch64/context.hpp` | Four new dispatch counters | ✓ EXISTS + SUBSTANTIVE | Counters added in the existing q4_* block of the struct. |
| Updated `tests/kernel/aarch64_tests.cpp` | Two new TEST_CASEs | ✓ EXISTS + SUBSTANTIVE | Numeric parity + dispatch counter assertions for q4_0 and q4_1. |

**Artifacts:** 10/10 verified

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `96-CONTEXT.md` and `96-01-PLAN.md` | Phase 96 ROADMAP success criteria | Goal-backward task mapping | ✓ WIRED | Each plan task maps explicitly to a success criterion (Tasks 3+4+5→SC1; Task 6→SC2; Task 5 (no f32 dequant)→SC3; Task 6+7 + counters→SC4). |
| `tests/kernel/aarch64_tests.cpp` new TEST_CASEs | New SML dispatch counters | `aarch64_sm` facade `process_event` + counter accessor methods | ✓ WIRED | Tests assert `optimized_q4_0_vector_dispatch_count() == 1` and `optimized_q4_1_vector_dispatch_count() == 1` after dispatching the matching `op_mul_mat` event. |
| New SML guards | Real NEON intrinsics | `can_use_neon_mul_mat_q4_*_vector` → `execute_neon_mul_mat_q4_*_vector_unchecked` → `dot_q4_*_q8_0_*_neon` | ✓ WIRED | Action functor calls `execute_neon_mul_mat_q4_*_vector_unchecked(ev.request)` directly; the unchecked path hits the new dot helpers in 4-row + 1-row tail loop. |
| Generic fall-through guard | Phase 96 SC2 | Updated `simd_op_mul_mat_generic` exclusions | ✓ WIRED | Generic fall-through now excludes both `simd_op_mul_mat_q4_0_vector` and `_q4_1_vector` so the generic backend cannot steal dispatch for the new lanes. |
| Focused doctest run | Verification confidence | `build/audit-native/emel_tests_bin '--test-case=*kernel_aarch64_q4_0_vector*,*kernel_aarch64_q4_1_vector*,*kernel_aarch64_q5_0_vector*,*kernel_aarch64_q8_0_vector*,*kernel_aarch64_q8_0*'` | ✓ WIRED | 7/7 cases passed, 118/118 assertions passed, 0 failures. |
| Broader regression sweep | Verification confidence | `--test-case='*kernel_aarch64*'` | ✓ WIRED | 56/56 aarch64 kernel cases passed, 7118 assertions all passing. |
| Whisper + kernel union | Verification confidence | `--test-case='*Whisper*,*model_whisper*,*EMEL Whisper sources*,*kernel*'` | ✓ WIRED | 123/123 cases passed, 13135 assertions all passing. |
| Scoped quality gate run | Verification confidence | `EMEL_QUALITY_GATES_CHANGED_FILES="<phase 96 changed files>" scripts/quality_gates.sh` | ✓ WIRED | Exit code 0. Coverage scoping limitation noted in SUMMARY (gcovr empty-report when changes are header-only inline code; gate exits 0 and the new code is exercised by the new doctest cases). |

**Wiring:** 8/8 connections verified

## Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| KERN-01 (narrowed): native EMEL kernel paths for `{q4_0, q4_1, q8_0}` | ✓ SATISFIED | - |
| KERN-02: no whole-tensor dequant-to-f32 fallback in quantized hot paths | ✓ SATISFIED | - |
| KERN-03: ARM optimized dispatch attribution + focused kernel tests | ✓ SATISFIED | - |

**Coverage:** 3/3 Phase 96 requirements satisfied (KERN-01 narrowed scope locked).

## Anti-Patterns Found

None within Phase 96 changes. All routing decisions live in `guards.hpp`/`sm.hpp` per
AGENTS.md; arithmetic and intrinsics live in the kernel actions layer; no generic
fall-through accidentally consumes the new dispatch lanes.

The discovered `is_native_quantized_dtype` predicate omission is corrected by the
narrowest possible change — adding q4_0 and q4_1 to the existing list of native
quantized dtypes (alongside q5_0, q8_0, and k-dtypes). This is a behavior-correct
predicate fix, not an anti-pattern introduction.

**Anti-patterns:** 0 found (0 blockers, 0 warnings)

## Human Verification Required

None — all phase must-haves were verified programmatically. The FULL NEON scope
decision was explicitly approved by agent-0 (`tf9ePyZNIHVRJYWD`) before
implementation began.

## Gaps Summary

**No Phase 96 gaps.** Phase goal achieved. Ready to proceed to Phase 97 (Whisper Audio
Frontend And Encoder) once the operator decides to advance.

**Known limitations carried forward:**
- Vanilla NEON (non-DOTPROD) widening fallback is preserved in source but not
  exercised on the current build host (which has `__ARM_FEATURE_DOTPROD`).
  Verifying it would require a non-DOTPROD build configuration; out of Phase 96
  scope.
- gcovr coverage scoping reports zero lines for header-only inline changes; tooling
  improvement deferred to a future planning/infrastructure phase.

**Unrelated residual risk** (carried from Phase 95):
- Pre-existing SIGSEGV in `tests/diarization/sortformer/modules/lifecycle_tests.cpp:74`
  (`sortformer modules bind maintained tensor contract`); reproducible at HEAD
  without Phase 96 edits; out of scope per agent-0 directive.

## Verification Metadata

**Verification approach:** Goal-backward (ROADMAP Phase 96 success criteria)
**Must-haves source:** `.planning/ROADMAP.md` Phase 96 success criteria
**Automated checks:** focused doctest (7/7), aarch64 sweep (56/56), Whisper+kernel
sweep (123/123), scoped quality gate (exit 0)
**Human checks required:** 0
**Total verification time:** ~50 min

---
*Verified: 2026-04-25T20:05:00Z*
*Verifier: autonomous-resume (worker)*
