# Phase 96: Native Quant Variant Kernels — Context

**Phase Goal:** Add native kernel-level support for the narrowed maintained Whisper quant
variant family `{q8_0, q4_0, q4_1}` (per Phase 95 narrowing approved 2026-04-25, agent-0
message `J2bGANuAk61CW6P6`), including ARM optimized dispatch proof.

**Requirements covered:** KERN-01 (narrowed family), KERN-02, KERN-03.

## Approved Scope (Option A — FULL NEON)

agent-0 message `tf9ePyZNIHVRJYWD` (2026-04-25): "FULL NEON, no exceptions. Implement
option A. Phase 96 must land real NEON intrinsics for q4_0 and q4_1 native quant kernels
and wire explicit SML dispatch/guard paths and focused tests. Do not use wrapper-only
scalar calls, attribution-only proof, whole-tensor dequant-to-f32 fallback, or defer
q4_0/q4_1 SIMD to Phase 101. Keep scope to the approved maintained family
{q8_0,q4_0,q4_1}."

## Current Kernel Landscape

For the narrowed `{q8_0, q4_0, q4_1}` family:

| Variant | Status |
|---------|--------|
| `q8_0` | Multiple real NEON paths exist in `src/emel/kernel/aarch64/actions.hpp` (`execute_neon_mul_mat_q8_0_vector_unchecked`, `_packed_bl4`, `_packed_bl8`, `_packed_bl8_full_groups`, `_packed_bl8_matrix_x4`); SML dispatch wired in `sm.hpp` and `guards.hpp`; dispatch counters in `context.hpp`. Already satisfies KERN-01/KERN-02/KERN-03 for q8_0. |
| `q4_0` | Only scalar `dot_q4_0_q8_0_row_scalar` in `src/emel/kernel/detail.hpp`. No NEON path, no SML dispatch lane, no dispatch counter. |
| `q4_1` | Only scalar `dot_q4_1_q8_0_row_scalar` in `src/emel/kernel/detail.hpp`. No NEON path, no SML dispatch lane, no dispatch counter. |

Reference pattern: `dot_q5_0_q8_0_row_neon` (real NEON intrinsics with `vdotq_s32` for
DOTPROD path, widening fallback for vanilla NEON), `dot_q5_0_q8_0_4rows_neon`,
`execute_neon_mul_mat_q5_0_vector_unchecked`, plus `simd_op_mul_mat_q5_0_vector` guard
and `exec_simd_op_mul_mat_q5_0_vector` action wired through the `sm.hpp` transition table.

## Block Layouts (Reference)

- `block_q4_0`: `fp16 d` + `qs[16]` packed bytes; each byte holds two signed 4-bit values
  recovered as `(qs[j] & 0x0f) - 8` (low half) and `(qs[j] >> 4) - 8` (high half). 32
  values per block (`QK4_0`).
- `block_q4_1`: `fp16 d` + `fp16 m` + `qs[16]` packed bytes; each byte holds two unsigned
  4-bit values; output is `qs * d + m`. 32 values per block (`QK4_1`).
- RHS for both is `block_q8_0` (`fp16 d` + `qs[32]` int8). Dot result for q4_0 is
  `sum(unpacked * rhs.qs) * lhs.d * rhs.d`. Dot result for q4_1 is
  `lhs.d * sum(unpacked * rhs.qs) * rhs.d + lhs.m * rhs.d * sum(rhs.qs)`.

## What Phase 96 Must Add

1. **Kernel layer (`src/emel/kernel/aarch64/actions.hpp`)** — real NEON intrinsics for
   `dot_q4_0_q8_0_row_neon`, `dot_q4_0_q8_0_4rows_neon`, `dot_q4_1_q8_0_row_neon`,
   `dot_q4_1_q8_0_4rows_neon` mirroring the q5_0 pattern. DOTPROD-fast path with widening
   fallback. No whole-tensor f32 dequant. q4_1 must accumulate the rhs sum required for
   the `lhs.m` contribution.
2. **Support fns** — `neon_q4_0_vector_supported`, `neon_q4_1_vector_supported`,
   `can_run_neon_mul_mat_q4_0_vector_request`, `can_use_neon_mul_mat_q4_0_vector`, and
   `q4_1` analogues.
3. **Execute fns** — `execute_neon_mul_mat_q4_0_vector_unchecked`,
   `execute_neon_mul_mat_q4_0_vector`, and `q4_1` analogues, mirroring the q5_0 vector
   path that quantizes the f32 src1 to q8_0 in a bounded stack array, then drives the
   4-row + 1-row tail loop.
4. **Dispatch counters** — extend `aarch64::action::context` with
   `optimized_q4_0_dispatch_count`, `optimized_q4_0_vector_dispatch_count`,
   `optimized_q4_1_dispatch_count`, `optimized_q4_1_vector_dispatch_count`. Add accessor
   methods on the SM facade.
5. **SML guard + action + transition** — `simd_op_mul_mat_q4_0_vector` and
   `simd_op_mul_mat_q4_1_vector` guards, paired action types, transitions in `sm.hpp`,
   and updated `simd_op_mul_mat_generic` fall-through to exclude the new vector lanes.
6. **Tests (`tests/kernel/aarch64_tests.cpp`)** — focused doctest mirroring
   `kernel_aarch64_q5_0_vector_route_is_explicit_and_numeric_match` for q4_0 and q4_1:
   builds synthetic blocks, runs scalar reference, runs NEON `execute_*_vector`, asserts
   numeric equivalence (epsilon-bounded), then drives the SML facade with the
   `op_mul_mat` event and asserts the new dispatch counters increment exactly once.
7. **Test helpers** — add `quantize_row_q4_0_ref` and `quantize_row_q4_1_ref` ref
   implementations in `src/emel/kernel/detail.hpp` (mirror the existing
   `quantize_row_q5_0_ref` pattern) so tests can construct deterministic blocks from
   dense rows. These helpers are kernel-owned data-plane code, not orchestration.

## Constraints

- AGENTS.md: kernel arithmetic and ARM intrinsics belong in `src/emel/kernel/**`. Routing
  decisions belong in SML `guards.hpp`/`sm.hpp`. The new NEON exec functions are bounded
  data-plane action code (no allocation, no orchestration).
- Hot-path allocation: the q5_0 vector path uses an `alignas(64) std::array<block_q8_0,
  MAX_Q8_0_BLOCKS>` on the stack; the new q4_0/q4_1 paths follow the same bounded stack
  pattern.
- Numeric parity: NEON path must match scalar path within a small epsilon (the reference
  q5_0 test uses `1.0e-5f`).
- DOTPROD/MATMUL_INT8 features: gate via `__ARM_FEATURE_DOTPROD` like the q5_0 reference.
  Keep a widening fallback for vanilla NEON.

## Definition Of Done

- New NEON intrinsics for q4_0 and q4_1 land in `src/emel/kernel/aarch64/actions.hpp`.
- New dispatch counters and accessors exist; SML guards, actions, and transitions exist.
- The `simd_op_mul_mat_generic` fall-through guard excludes the new q4_0/q4_1 vector lanes.
- Focused doctest filter
  `--test-case='*kernel_aarch64_q4_0_vector*,*kernel_aarch64_q4_1_vector*,*kernel_aarch64_q5_0_vector*,*kernel_aarch64_q8_0*'`
  passes with numeric parity + dispatch counter assertions.
- Scoped `EMEL_QUALITY_GATES_CHANGED_FILES="<phase 96 changed files>" scripts/quality_gates.sh`
  exits 0.
- 96-01-SUMMARY.md and 96-VERIFICATION.md written; STATE.md/ROADMAP.md updated for Phase 96
  closeout.
