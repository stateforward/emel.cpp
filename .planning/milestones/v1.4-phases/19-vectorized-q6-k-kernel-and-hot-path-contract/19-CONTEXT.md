# Phase 19: Vectorized q6_K Kernel And Hot-Path Contract - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 19 completes the maintained AArch64 quantized row-kernel set by cutting `q6_K x q8_K`
over to an EMEL-owned vectorized row helper, then locks the maintained optimized quantized hot
path to the same effective operand class with zero-allocation dispatch behavior for supported
`q2_K/q3_K/q6_K` requests.

This phase stays inside the existing backend and maintained test surfaces. It does not change
Boost.SML structure, public APIs, runtime wrappers, or benchmark/parity publication surfaces.

</domain>

<decisions>
## Implementation Decisions

### Kernel Seam
- Keep the `q6_K` cutover inside `src/emel/kernel/aarch64/actions.hpp`, specifically the
  quantized branch of `execute_neon_mul_mat(...)`; do not add transition rows, events, or new
  runtime wrappers in this phase.
- Preserve the existing request validation, row-byte layout, and per-column `q8_K` staging that
  the quantized AArch64 matmul path already performs.
- Replace only the maintained `q6_K x q8_K` row-dot execution path in Phase 19; q2 and q3
  behavior should remain as landed in Phases 17 and 18.

### Operand And Performance Contract
- Reuse the existing `block_q8_k` operand pipeline and the backend-local
  `dot_q6_k_q8_k_block_neon(...)` arithmetic that already exists in the AArch64 backend.
- Preserve zero-allocation dispatch by keeping all hot-path scratch bounded to existing fixed
  storage; do not introduce heap-backed staging or dequantize-to-f32 substitution.
- Keep the effective operand class fixed to quantized `q*_K x q8_K`; Phase 19 proof must show the
  supported optimized path no longer depends on the shared scalar row helpers for q2/q3/q6.

### Proof Surface
- Add failing-first proof at the maintained kernel seam before landing the vectorized q6 cutover.
- Keep Phase 19 proof on maintained kernel-facing surfaces such as
  `tests/kernel/aarch64_tests.cpp` and `tests/kernel/lifecycle_tests.cpp`; do not widen generator,
  paritychecker, or benchmark publication in this phase.
- Reuse the existing test-binary allocation tracker rather than inventing a new runtime hook so
  kernel dispatch can prove zero-allocation behavior directly.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/kernel/aarch64/actions.hpp` already owns the AArch64 quantized matmul seam and already
  contains backend-local `dot_q6_k_q8_k_block_neon(...)` arithmetic.
- Phases 17 and 18 already established the maintained seam pattern for q2 and q3 via
  backend-local row helper wiring plus backend-local path attribution counters.
- `tests/graph/graph_tests.cpp` already installs a process-wide allocation tracker through global
  `operator new/delete`, so Phase 19 can reuse that tracker from kernel tests instead of creating
  a second allocator hook.

### Established Patterns
- Backend specialization lives in backend-local helpers and wrapper accessors while SML structure
  stays unchanged.
- Supported optimized quantized requests are proven through backend-local optimized/shared counters
  rather than widened runtime wrappers.
- Hot-path numeric work stays allocation-free and bounded inside action/detail helpers.

### Integration Points
- `src/emel/kernel/aarch64/actions.hpp` is the exact seam where quantized matmul currently stages
  `q8_K` blocks and still dispatches `q6_K` through the shared scalar row helper.
- `src/emel/kernel/detail.hpp` remains the scalar parity reference for q6 row results and operand
  layout.
- `tests/kernel/aarch64_tests.cpp` is the narrowest maintained proof surface for direct q6 kernel
  equivalence plus alloc-free quantized dispatch proof.

</code_context>

<specifics>
## Specific Ideas

- Introduce a backend-local q6 row helper that accumulates over the staged `block_q8_k` array via
  `dot_q6_k_q8_k_block_neon(...)` and mirrors the scalar row contract exactly.
- Add q6-focused tests that fail first until the maintained AArch64 path can distinguish
  vectorized q6 execution from the scalar helper on the canonical operand layout.
- Expose q6 optimized/shared counters at the backend seam, then add an alloc-free proof that runs
  supported q2/q3/q6 requests under the existing global allocation tracker and asserts zero
  allocations plus zero shared-fallback claims on supported AArch64 execution.

</specifics>

<deferred>
## Deferred Ideas

- Runtime-chain proof and publication remain Phase 20.
- Parity publication and maintained benchmark attribution remain Phases 20 and 21.

</deferred>

---
*Phase: 19-vectorized-q6-k-kernel-and-hot-path-contract*
*Context gathered: 2026-03-22*
