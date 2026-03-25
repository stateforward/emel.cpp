# Phase 19: Vectorized q6_K Kernel And Hot-Path Contract - Research

**Researched:** 2026-03-22
**Domain:** EMEL AArch64 backend-local q6_K vectorized quantized matmul cutover plus alloc-free
hot-path proof on the canonical ARM Llama-68M path
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
## Phase Boundary

Phase 19 completes the maintained AArch64 quantized row-kernel set by cutting `q6_K x q8_K`
over to an EMEL-owned vectorized row helper, then locks the maintained optimized quantized hot
path to the same effective operand class with zero-allocation dispatch behavior for supported
`q2_K/q3_K/q6_K` requests.

## Implementation Decisions

### Kernel Seam
- Keep the `q6_K` cutover inside `src/emel/kernel/aarch64/actions.hpp`, specifically the
  quantized branch of `execute_neon_mul_mat(...)`; do not add transition rows, events, or new
  runtime wrappers in this phase.
- Preserve the existing request validation, row-byte layout, and per-column `q8_K` staging that
  the quantized AArch64 matmul path already performs.
- Replace only the maintained `q6_K x q8_K` row-dot execution path in Phase 19.

### Proof Surface
- Add failing-first proof at the maintained kernel seam before landing the vectorized q6 cutover.
- Keep proof on kernel-facing maintained surfaces only; do not widen paritychecker or benchmark
  publication in this phase.
- Reuse existing test allocation tracking instead of adding a production hook.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PORT-06 | The canonical Llama-68M ARM generation slice can execute `q6_K x q8_K` hot-path dot products through an EMEL-owned vectorized AArch64 kernel instead of the current scalar row helper. | `src/emel/kernel/aarch64/actions.hpp` already has `dot_q6_k_q8_k_block_neon(...)`; Phase 19 only needs a backend-local row helper plus wiring from the existing quantized `execute_neon_mul_mat(...)` seam. |
| PORT-07 | The vectorized quantized kernels preserve zero-allocation hot-path behavior, keep the same effective operand class, and avoid dequantize-to-f32 fallback on supported requests. | The current quantized seam already stages `block_q8_k` into a fixed `std::array` and branches by dtype. With q6 cut over and q6/q2/q3 path counters exposed at the backend seam, a test can prove supported requests stay on the optimized quantized path and incur zero global allocations under the existing test-binary allocation tracker. |
| ARCH-02 (phase-local guardrail) | The optimization stays a data-plane replacement inside the current runtime chain and does not rewrite actor structure. | Phases 17 and 18 already proved the seam can change at backend-helper level only. No SML changes are needed if Phase 19 stays inside backend-local helpers, counters, and tests. |
</phase_requirements>

## Summary

The current AArch64 quantized matmul seam is specialized for q2 and q3, but the maintained q6
path still falls back to `src/emel/kernel/detail.hpp::dot_q6_k_q8_k_row_scalar(...)` even though
`src/emel/kernel/aarch64/actions.hpp` already contains backend-local q6 block NEON arithmetic.
That makes the first half of Phase 19 another narrow dtype-local cutover:

1. Add a backend-local `dot_q6_k_q8_k_row_neon(...)` helper that accumulates staged q8 blocks with
   `dot_q6_k_q8_k_block_neon(...)` and mirrors the scalar row contract exactly.
2. Wire only the `dtype_q6_k` branch in `execute_neon_mul_mat(...)` to the new row helper.
3. Add q6-focused kernel tests and backend-local q6 path attribution so supported q6 requests are
   distinguishable from the previous shared scalar helper.

The second half of Phase 19 is proof, not architecture change. The quantized seam already uses a
fixed-size `std::array<block_q8_k, MAX_Q8_K_BLOCKS>` and has no heap-backed staging in the
supported optimized path. The cleanest proof is to expose the existing test-binary allocation
tracker from `tests/graph/graph_tests.cpp` through a shared test helper, then run supported
q2/q3/q6 backend dispatch under that tracker and assert:

1. allocation count remains zero
2. optimized counters increment for q2/q3/q6 on supported AArch64 execution
3. shared counters remain zero, proving the supported path no longer depends on the scalar row
   helpers

This satisfies Phase 19 without widening `kernel::any`, generator wrappers, paritychecker, or
bench.

## Likely File Changes

| File | Why |
|------|-----|
| `src/emel/kernel/aarch64/actions.hpp` | Exact q6 cutover seam; needs a q6 row helper, q6 branch wiring, and q6 attribution increments. |
| `src/emel/kernel/aarch64/context.hpp` | Narrowest place to hold backend-local q6 path attribution counters. |
| `src/emel/kernel/aarch64/sm.hpp` | Backend-local accessor surface for q6 attribution used by kernel tests only. |
| `tests/kernel/aarch64_tests.cpp` | Primary proof surface for q6 scalar-equivalence, q6 path attribution, and alloc-free quantized dispatch proof. |
| `tests/kernel/lifecycle_tests.cpp` | Secondary maintained proof surface for q6 backend dispatch acceptance and attribution. |
| `tests/graph/graph_tests.cpp` | Must share the existing allocation tracker instead of hiding it in an anonymous namespace. |
| `tests/allocation_tracker.hpp` | Shared test-only helper exposing allocation scope to multiple test translation units. |

## Architecture Patterns

### Pattern 1: Complete The Existing Quantized Row Set
Finish q6 with the same backend-local row-helper pattern used for q2 and q3. Do not invent a new
runtime path for the final dtype.

### Pattern 2: Prove The Contract At The Backend Seam
Use backend-local optimized/shared counters plus alloc-free tests to prove the maintained hot-path
contract. Do not widen public or runtime surfaces in Phase 19.

### Pattern 3: Reuse Existing Test Infrastructure
The binary already has allocation tracking via global `operator new/delete`. Exposing it through a
header is lower-risk than introducing a second allocator override or production counter.

## Anti-Patterns To Avoid

- Do not change `sm.hpp` transition rows or event shapes.
- Do not replace q6 with dequantize-to-f32 or tool-only fallbacks.
- Do not widen proof into paritychecker or benchmark publication in this phase.
- Do not make q6 attribution ambiguous by overloading q2/q3 counters.
- Do not add new heap-backed scratch for q8 staging or row execution.

## Common Pitfalls

### Pitfall 1: Forgetting The Supported-Path Contract Is Cross-Dtype
Phase 19 is not just q6 arithmetic. It must prove the supported optimized quantized path is closed
for q2/q3/q6 together.

### Pitfall 2: Reusing Anonymous Test Allocation State
`tests/graph/graph_tests.cpp` currently hides allocation tracking in an anonymous namespace. Kernel
tests cannot reuse it until that state is surfaced through shared test-only declarations.

### Pitfall 3: Mistaking Shared Scalar Quantized Fallback For Acceptable Final State
After Phase 19, supported AArch64 q2/q3/q6 requests should not depend on shared scalar row
helpers. Shared fallback remains valid only for unsupported or forced-off execution.

---
*Phase: 19-vectorized-q6-k-kernel-and-hot-path-contract*
*Research completed: 2026-03-22*
