# Phase 17: Vectorized q2_K Kernel - Research

**Researched:** 2026-03-22
**Domain:** EMEL AArch64 backend-local q2_K vectorized quantized matmul cutover on the canonical
ARM Llama-68M path
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
## Phase Boundary

Phase 17 delivers an EMEL-owned vectorized AArch64 `q2_K x q8_K` hot-path kernel inside the
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
- Replace only the maintained `q2_K x q8_K` row-dot execution path in Phase 17; `q3_K` and `q6_K`
  remain on their current helpers until Phases 18 and 19.

### Proof Surface
- Add failing-first proof at the maintained kernel seam before landing the vectorized q2 cutover.
- Keep proof on kernel-facing maintained surfaces only; do not widen paritychecker or benchmark
  publication in this phase.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PORT-04 | The canonical Llama-68M ARM generation slice can execute `q2_K x q8_K` hot-path dot products through an EMEL-owned vectorized AArch64 kernel instead of the current scalar row helper. | `src/emel/kernel/aarch64/actions.hpp` already has `dot_q2_k_q8_k_block_neon(...)`; Phase 17 only needs a backend-local row helper plus wiring from the existing quantized `execute_neon_mul_mat(...)` seam. |
| ARCH-02 (phase-local guardrail) | The optimization stays a data-plane replacement inside the current runtime chain and does not rewrite actor structure. | The AArch64 backend already owns the quantized matmul dispatch row. No SML changes are needed if the q2 cutover stays inside backend-local helpers and optional backend-local observability. |
</phase_requirements>

## Summary

The current AArch64 quantized matmul seam is already partially specialized, but the maintained q2
path still falls back to a shared scalar row helper. `execute_neon_mul_mat(...)` validates the
request, re-quantizes `src1` into `block_q8_k` tiles, and then branches by `src0_type`. For
`q2_K`, `q3_K`, and `q6_K` it still calls the scalar row helpers from `src/emel/kernel/detail.hpp`
even though `src/emel/kernel/aarch64/actions.hpp` already contains backend-local block NEON
arithmetic for all three dtypes.

That makes Phase 17 a narrow cutover. The safe implementation is:

1. Add a backend-local `dot_q2_k_q8_k_row_neon(...)` helper that accumulates the staged q8 blocks
   with the existing `dot_q2_k_q8_k_block_neon(...)` arithmetic and mirrors the scalar row
   contract exactly.
2. Wire only the `dtype_q2_k` branch in `execute_neon_mul_mat(...)` to the new row helper.
3. Add failing-first q2-focused kernel tests, then backend-local q2 path attribution so the kernel
   seam can prove supported q2 requests used the vectorized path.

The recommended proof stays backend-local. AArch64 already carries backend-only flash counters in
`action::context` and `aarch64::sm`. Reusing that pattern for q2 path attribution is narrower than
extending `kernel::any` or `generator::sm`, which belongs to Phase 20 runtime proof.

## Likely File Changes

| File | Why |
|------|-----|
| `src/emel/kernel/aarch64/actions.hpp` | Exact q2 cutover seam; needs a row helper and q2 branch wiring. |
| `src/emel/kernel/aarch64/context.hpp` | Narrowest place to hold backend-local q2 path attribution counters if seam proof needs observability. |
| `src/emel/kernel/aarch64/sm.hpp` | Optional backend-local accessor surface for q2 path attribution used by kernel tests only. |
| `tests/kernel/aarch64_tests.cpp` | Primary proof surface for q2 scalar-equivalence and supported vectorized-path attribution. |
| `tests/kernel/lifecycle_tests.cpp` | Secondary maintained proof surface for backend dispatch acceptance and q2 canonical behavior. |

## Architecture Patterns

### Pattern 1: Shared Validation, Backend-Local Row Kernel
Keep request validation and q8 staging exactly where they are now. Replace only the q2 row-dot
math beneath the existing AArch64 quantized branch.

### Pattern 2: Backend-Local Attribution Only
If path-selection proof needs counters, add them only to `aarch64::action::context` and
`aarch64::sm`. Do not forward them into `kernel::any`, generator wrappers, paritychecker, or bench
in Phase 17.

### Pattern 3: Failing-First Kernel-Seam Tests
Write q2-focused doctests that fail first until the AArch64 backend exposes a distinguishable
vectorized q2 path and preserves scalar-equivalent numeric results.

## Anti-Patterns To Avoid

- Do not change `sm.hpp` transition rows or event shapes.
- Do not replace the q2 hot path with dequantize-to-f32 or tool-only fallbacks.
- Do not widen proof into paritychecker or benchmark publication in this phase.
- Do not accidentally cut over q3 or q6 behavior early unless a tiny shared refactor is needed.
- Do not add queueing, state flags, or orchestration branching for a pure numeric helper swap.

## Common Pitfalls

### Pitfall 1: Proving NEON Arithmetic Without Proving Seam Adoption
Direct block-helper tests alone do not prove `execute_neon_mul_mat(...)` stopped using the scalar
row helper. Add a backend-local seam signal or equivalent test-visible attribution.

### Pitfall 2: Mixing q2 Work With q3/q6 Cleanup
The block helpers are adjacent and similar, but Phase 17 only closes q2. Keep q3 and q6 changes
strictly limited to harmless shared refactors.

### Pitfall 3: Breaking The Operand Contract
The maintained hot path is the exact q8 staging and quantized-row layout already shipped. Any
alternate operand pipeline would weaken truthful parity and benchmark claims later in the
milestone.

---
*Phase: 17-vectorized-q2-k-kernel*
*Research completed: 2026-03-22*
