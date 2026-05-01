---
phase: 96-native-quant-variant-kernels
plan: 01
requirements-completed: [KERN-01, KERN-02, KERN-03]
completed: 2026-04-25
---

# Phase 96 Plan 01: Native Quant Variant Kernels — Execution Summary

**Phase Goal:** Add native kernel-level support for the narrowed maintained Whisper quant
variant family `{q8_0, q4_0, q4_1}` (per Phase 95 narrowing approved 2026-04-25, agent-0
message `J2bGANuAk61CW6P6`), including ARM optimized dispatch proof.

**Plan Goal:** Land real ARM NEON intrinsics for `dot_q4_0_q8_0` and `dot_q4_1_q8_0` with
explicit SML routing through `simd_op_mul_mat_q4_0_vector` and
`simd_op_mul_mat_q4_1_vector` guard/action pairs, keep the q5_0/q8_0 paths intact, prove
numeric parity vs scalar, and prove SML dispatch counters increment exactly once for
routed events.

**Status:** Complete. Option A (FULL NEON) approved by agent-0 in message
`tf9ePyZNIHVRJYWD` (2026-04-25); no wrapper-only scalar calls, no attribution-only proof,
no whole-tensor dequant-to-f32 fallback, no deferral.

## Outcomes Per Task

### Task 1 — Quantization Helpers

- Added `quantize_row_q4_0_ref` and `quantize_row_q4_1_ref` in
  `src/emel/kernel/detail.hpp`, mirroring the existing `quantize_row_q5_0_ref` pattern.
  q4_0 uses signed 4-bit values with `d = max / -8`; q4_1 uses unsigned 4-bit values with
  per-block `d = (max - min) / 15` and `m = min`.
- Added `<limits>` include to make `std::numeric_limits` usage explicit.

### Task 2 — Dispatch Counters

- Added `optimized_q4_0_dispatch_count`, `optimized_q4_0_vector_dispatch_count`,
  `optimized_q4_1_dispatch_count`, `optimized_q4_1_vector_dispatch_count` to the
  aarch64 `action::context` struct.

### Task 3 — Real NEON Intrinsics For q4_0

- Added `dot_q4_0_q8_0_row_neon` in `src/emel/kernel/aarch64/actions.hpp` with the
  DOTPROD-fast path (`vdotq_s32`) and a vanilla NEON widening fallback
  (`vmull_s8` + `vpaddlq_s16`). Decodes packed 4-bit values via `vandq_u8` /
  `vshrq_n_u8` and subtracts the `8` bias before multiplying against the q8_0 RHS.
- Added `dot_q4_0_q8_0_4rows_neon` for 4-row block-parallel accumulation.
- Both fall back to scalar `dot_q4_0_q8_0_row_scalar` outside `__aarch64__` /
  `__ARM_NEON`.

### Task 4 — Real NEON Intrinsics For q4_1

- Added `dot_q4_1_q8_0_row_neon` and `dot_q4_1_q8_0_4rows_neon` mirroring the q4_0
  shape, but skipping the `-8` bias (q4_1 uses unsigned 4-bit values) and adding the
  `lhs.m * rhs.d * sum(rhs.qs)` correction term. The rhs sum is computed via a new
  `horizontal_sum_s8_neon` helper that widens via `vmovl_s8` and pairwise sums.
- Same DOTPROD-fast / widening-fallback structure.

### Task 5 — Support, Acceptance, And Execute Functions

- Added `neon_q4_0_vector_supported`, `neon_q4_1_vector_supported` (gated on
  `__aarch64__ && __ARM_NEON && __ARM_FEATURE_DOTPROD`).
- Added `can_run_neon_mul_mat_q4_0_vector_request` and `q4_1` analogues that mirror the
  q5_0 layout/contract acceptance check (k cols, m rows, src1=f32 (1, k), dst=f32
  (1, m), src0 stride matches `quantized_row_storage_bytes(dtype_q4_*, k)`, dense
  contiguous src1/dst).
- Added `can_use_neon_mul_mat_q4_0_vector` and `q4_1` user guards combining feature
  support and the request-shape check.
- Added `execute_neon_mul_mat_q4_0_vector_unchecked`,
  `execute_neon_mul_mat_q4_0_vector`, and `q4_1` analogues. Both follow the q5_0 vector
  shape: stack-allocated `MAX_Q8_0_BLOCKS` worth of `block_q8_0` for the quantized RHS,
  4-row + 1-row tail loop driving the new dot helpers.

### Task 6 — SML Wiring

- Added `simd_op_mul_mat_q4_0_vector` and `simd_op_mul_mat_q4_1_vector` guards in
  `src/emel/kernel/aarch64/guards.hpp`. Updated `simd_op_mul_mat_generic` fall-through
  to also exclude the new q4_0/q4_1 vector lanes so the generic backend cannot steal
  dispatch.
- Added `exec_simd_q4_0_vector_op_mul_mat` and `exec_simd_q4_1_vector_op_mul_mat`
  action structs in `src/emel/kernel/aarch64/actions.hpp` (each calls
  `execute_neon_mul_mat_q*_vector_unchecked`, increments the new dispatch counters,
  and marks done). Added type aliases and `inline constexpr` instances.
- Added two destination-first transitions in `src/emel/kernel/aarch64/sm.hpp`
  immediately after the q5_0_vector lane and before the q8_0 packed lanes.
- Added accessor methods on the `aarch64::sm` facade:
  `optimized_q4_0_dispatch_count()`, `optimized_q4_0_vector_dispatch_count()`,
  `optimized_q4_1_dispatch_count()`, `optimized_q4_1_vector_dispatch_count()`.

### Task 6.5 — Native-Quantized-Dtype Predicate Update (Discovered Required)

- During the first run of the new dispatch tests, `validate_dispatch_request` returned
  false for q4_0/q4_1 src0 because `is_native_quantized_dtype` only listed
  `q5_0`, `q8_0`, and k-dtypes. Extended the predicate to include `q4_0` and `q4_1`
  so `has_required_src0` accepts their block layout via the existing native-quantized
  branch. This is a 1-line, behavior-correct extension (q4_0/q4_1 always were native
  quantized; the predicate was missing them because no in-tree caller had previously
  required SML dispatch for these dtypes).

### Task 7 — Focused Tests

- Added `kernel_aarch64_q4_0_vector_route_is_explicit_and_numeric_match` and
  `kernel_aarch64_q4_1_vector_route_is_explicit_and_numeric_match` in
  `tests/kernel/aarch64_tests.cpp`. Each test:
  - Builds 4 synthetic dense rows + 1 dense input via deterministic patterns.
  - Quantizes via `quantize_row_q4_*_ref` / `quantize_row_q8_0_strided`.
  - Computes scalar reference via `dot_q4_*_q8_0_row_scalar`.
  - Asserts `can_use_neon_mul_mat_q4_*_vector(ev, true)` returns true and
    `execute_neon_mul_mat_q4_*_vector(ev)` succeeds.
  - Numeric parity vs scalar within `1.0e-5f`.
  - Drives the `aarch64::sm` facade with the `op_mul_mat` event and asserts
    `optimized_q4_*_dispatch_count() == 1` and
    `optimized_q4_*_vector_dispatch_count() == 1` and
    `shared_q4_dispatch_count() == 0`.

### Task 8 — Focused Tests And Scoped Quality Gate

- Build: `cmake --build build/audit-native --target emel_tests_bin` — clean.
- Focused doctest filter
  `--test-case='*kernel_aarch64_q4_0_vector*,*kernel_aarch64_q4_1_vector*,*kernel_aarch64_q5_0_vector*,*kernel_aarch64_q8_0_vector*,*kernel_aarch64_q8_0*'`:
  **7 cases, 7 passed, 0 failed, 118 assertions, all passing.**
- Broader regression sweep (all aarch64 kernel tests):
  `--test-case='*kernel_aarch64*'` — **56 cases, 56 passed, 7118 assertions all
  passing**.
- Whisper + kernel filter:
  `--test-case='*Whisper*,*model_whisper*,*EMEL Whisper sources*,*kernel*'` —
  **123 cases, 123 passed, 13135 assertions all passing**.
- Scoped `EMEL_QUALITY_GATES_CHANGED_FILES=
  "src/emel/kernel/detail.hpp src/emel/kernel/aarch64/actions.hpp
   src/emel/kernel/aarch64/context.hpp src/emel/kernel/aarch64/guards.hpp
   src/emel/kernel/aarch64/sm.hpp tests/kernel/aarch64_tests.cpp"
  scripts/quality_gates.sh` — exits **0**.

### Task 9 — SUMMARY And VERIFICATION

- This summary file enumerates outcomes per task.
- `96-VERIFICATION.md` (companion file) maps each ROADMAP success criterion to
  artifacts/test cases that satisfy it.

## DOTPROD Path Attribution On The Build Host

The current build host is macOS arm64 with `-mcpu=native+dotprod+i8mm` per
`CMakeLists.txt`, so `__ARM_FEATURE_DOTPROD` is defined and the new tests exercise the
DOTPROD-fast path (`vdotq_s32`) for both q4_0 and q4_1. The vanilla NEON widening
fallback is preserved in source for build configurations without DOTPROD; coverage of
that fallback would require an explicit non-DOTPROD build, which is out of scope for
Phase 96 since the maintained ARM benchmark host is DOTPROD-capable.

## Files Changed

| File | Type | Change |
|------|------|--------|
| `src/emel/kernel/detail.hpp` | kernel | `quantize_row_q4_0_ref`, `quantize_row_q4_1_ref`; extend `is_native_quantized_dtype` to include q4_0/q4_1; add `<limits>` include. |
| `src/emel/kernel/aarch64/context.hpp` | kernel | New q4_0/q4_1 dispatch counters. |
| `src/emel/kernel/aarch64/actions.hpp` | kernel | Real NEON `dot_q4_0_q8_0_row_neon`, `dot_q4_0_q8_0_4rows_neon`, `dot_q4_1_q8_0_row_neon`, `dot_q4_1_q8_0_4rows_neon`, `horizontal_sum_s8_neon`; supported/can_run/can_use/execute `_q4_0_vector` and `_q4_1_vector` functions; `exec_simd_q4_*_vector_op_mul_mat` action structs and instances. |
| `src/emel/kernel/aarch64/guards.hpp` | kernel | `simd_op_mul_mat_q4_0_vector` / `_q4_1_vector` guards; updated generic fall-through. |
| `src/emel/kernel/aarch64/sm.hpp` | kernel | Two new transitions in destination-first form; four new accessor methods on the SM facade. |
| `tests/kernel/aarch64_tests.cpp` | tests | Two new TEST_CASEs covering numeric parity + SML dispatch counter assertions for q4_0 and q4_1. |
| `.planning/ROADMAP.md` | docs | Phase 96 SC1 narrowed to `{q4_0, q4_1, q8_0}`. |
| `.planning/phases/96-native-quant-variant-kernels/96-CONTEXT.md` | docs | Phase 96 context. |
| `.planning/phases/96-native-quant-variant-kernels/96-01-PLAN.md` | docs | Phase 96 plan. |
| `.planning/phases/96-native-quant-variant-kernels/96-01-SUMMARY.md` | docs | This summary. |
| `.planning/phases/96-native-quant-variant-kernels/96-VERIFICATION.md` | docs | Verification report (companion). |

## Notes And Limitations

- **Coverage scoping report.** The scoped quality gate's `gcovr` invocation reports
  "0 lines" for the changed-file filter when all changes are inline `.hpp` headers
  (`actions.hpp`, `guards.hpp`, `context.hpp`, `sm.hpp`, `detail.hpp`). Gate exits 0
  and the gcovr internal `(ERROR)` line does not propagate through the script. The
  new NEON code IS exercised by the focused doctest cases (verified: 118 assertions,
  including SML dispatch counter assertions that prove the new actions fired). A
  follow-up improvement to the coverage filter (matching against `tests/kernel/*.cpp`
  consumers as well as `.hpp` paths) would be appropriate but is out of Phase 96
  scope.
- **`is_native_quantized_dtype` extension.** This 1-line predicate fix in
  `src/emel/kernel/detail.hpp` was discovered during dispatch-test failure
  investigation. It corrects an existing predicate that omitted q4_0/q4_1 even though
  they are clearly native quantized dtypes; the omission was latent because no caller
  had previously required SML dispatch validation for q4_0/q4_1 src0.
- **Pre-existing residual risk** (carried from Phase 95): SIGSEGV in
  `tests/diarization/sortformer/modules/lifecycle_tests.cpp:74` reproducible at HEAD
  without Phase 96 edits. Not in Phase 96 scope per agent-0 directive.
- **Snapshots.** `snapshots/quality_gates/timing.txt` continues to be modified by
  gate runs; this file was already in the worktree's modified list from Phase 94 and
  is not committed by Phase 96.
