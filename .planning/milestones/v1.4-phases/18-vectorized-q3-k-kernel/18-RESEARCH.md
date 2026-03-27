# Phase 18: Vectorized q3_K Kernel - Research

**Researched:** 2026-03-22
**Domain:** EMEL AArch64 backend-local q3_K vectorized quantized matmul cutover on the canonical
ARM Llama-68M path
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
## Phase Boundary

Phase 18 delivers an EMEL-owned vectorized AArch64 `q3_K x q8_K` hot-path kernel inside the
existing kernel backend seam for the canonical Llama-68M ARM workload. This phase stays
data-plane-only: it does not change Boost.SML structure, public APIs, acceptance surfaces, or the
effective operand contract already shipped through the generator -> graph -> processor -> kernel
chain.

## Implementation Decisions

### Kernel Seam
- Keep the cutover inside `src/emel/kernel/aarch64/actions.hpp`, specifically the quantized
  branch of `execute_neon_mul_mat(...)`; do not add transition rows, events, or new runtime
  wrappers in this phase.
- Preserve the existing request validation, row-byte layout, and per-column `q8_K` staging that
  `execute_neon_mul_mat(...)` already performs for quantized matmul.
- Replace only the maintained `q3_K x q8_K` row-dot execution path in Phase 18; `q6_K` remains on
  its current helper until Phase 19.

### Proof Surface
- Add failing-first proof at the maintained kernel seam before landing the vectorized q3 cutover.
- Keep proof on kernel-facing maintained surfaces only; do not widen paritychecker or benchmark
  publication in this phase.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PORT-05 | The canonical Llama-68M ARM generation slice can execute `q3_K x q8_K` hot-path dot products through an EMEL-owned vectorized AArch64 kernel instead of the current scalar row helper. | `src/emel/kernel/aarch64/actions.hpp` already has `dot_q3_k_q8_k_block_neon(...)`; Phase 18 only needs a backend-local row helper plus wiring from the existing quantized `execute_neon_mul_mat(...)` seam. |
| ARCH-02 (phase-local guardrail) | The optimization stays a data-plane replacement inside the current runtime chain and does not rewrite actor structure. | Phase 17 already proved the seam can change at backend-helper level only. No SML changes are needed if the q3 cutover stays inside backend-local helpers and backend-local observability. |
</phase_requirements>

## Summary

The current AArch64 quantized matmul seam is now specialized for q2, but the maintained q3 path
still falls back to a shared scalar row helper. `execute_neon_mul_mat(...)` validates the request,
re-quantizes `src1` into `block_q8_k` tiles, and branches by `src0_type`. The `q3_K` branch still
calls `src/emel/kernel/detail.hpp::dot_q3_k_q8_k_row_scalar(...)` even though
`src/emel/kernel/aarch64/actions.hpp` already contains backend-local q3 block NEON arithmetic.

That makes Phase 18 another narrow cutover. The safe implementation is:

1. Add a backend-local `dot_q3_k_q8_k_row_neon(...)` helper that accumulates the staged q8 blocks
   with the existing `dot_q3_k_q8_k_block_neon(...)` arithmetic and mirrors the scalar row
   contract exactly.
2. Wire only the `dtype_q3_k` branch in `execute_neon_mul_mat(...)` to the new row helper.
3. Add failing-first q3-focused kernel tests, then backend-local q3 path attribution so the kernel
   seam can prove supported q3 requests used the vectorized path.

The recommended proof stays backend-local. Phase 17 already established the pattern with q2-only
context counters and `aarch64::sm` accessors, and Phase 18 should mirror that rather than widening
`kernel::any`, `generator::sm`, paritychecker, or bench.

## Likely File Changes

| File | Why |
|------|-----|
| `src/emel/kernel/aarch64/actions.hpp` | Exact q3 cutover seam; needs a row helper and q3 branch wiring. |
| `src/emel/kernel/aarch64/context.hpp` | Narrowest place to hold backend-local q3 path attribution counters if seam proof needs observability. |
| `src/emel/kernel/aarch64/sm.hpp` | Optional backend-local accessor surface for q3 path attribution used by kernel tests only. |
| `tests/kernel/aarch64_tests.cpp` | Primary proof surface for q3 scalar-equivalence and supported vectorized-path attribution. |
| `tests/kernel/lifecycle_tests.cpp` | Secondary maintained proof surface for backend dispatch acceptance and q3 canonical behavior. |

## Architecture Patterns

### Pattern 1: Shared Validation, Backend-Local Row Kernel
Keep request validation and q8 staging exactly where they are now. Replace only the q3 row-dot
math beneath the existing AArch64 quantized branch.

### Pattern 2: Backend-Local Attribution Only
If path-selection proof needs counters, add them only to `aarch64::action::context` and
`aarch64::sm`. Do not forward them into `kernel::any`, generator wrappers, paritychecker, or bench
in Phase 18.

### Pattern 3: Mirror The Proven Phase 17 Seam Pattern
Reuse the q2 cutover structure: direct row-helper equivalence test, backend-local optimized/shared
dispatch counts, and lifecycle proof at the backend seam.

## Anti-Patterns To Avoid

- Do not change `sm.hpp` transition rows or event shapes.
- Do not replace the q3 hot path with dequantize-to-f32 or tool-only fallbacks.
- Do not widen proof into paritychecker or benchmark publication in this phase.
- Do not accidentally cut over q6 behavior early unless a tiny shared refactor is needed.
- Do not add queueing, state flags, or orchestration branching for a pure numeric helper swap.

## Common Pitfalls

### Pitfall 1: Reusing q2 Attribution Names Instead Of Adding q3-Specific Proof
Phase 17 introduced q2-only seam counters. Phase 18 needs its own q3-specific proof rather than
overloading the q2 signals and making future interpretation ambiguous.

### Pitfall 2: Missing Multi-Block q3 Coverage
The q3 block arithmetic is more involved than q2 because of packed sign/high-bit handling. Add a
row-level q3 equivalence test, not just branch wiring.

### Pitfall 3: Breaking The Operand Contract
The maintained hot path is the exact q8 staging and quantized-row layout already shipped. Any
alternate operand pipeline would weaken truthful parity and benchmark claims later in the
milestone.

---
*Phase: 18-vectorized-q3-k-kernel*
*Research completed: 2026-03-22*
