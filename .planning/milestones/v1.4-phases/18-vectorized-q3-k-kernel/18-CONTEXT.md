# Phase 18: Vectorized q3_K Kernel - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 18 delivers an EMEL-owned vectorized AArch64 `q3_K x q8_K` hot-path kernel inside the
existing kernel backend seam for the canonical Llama-68M ARM workload. This phase stays
data-plane-only: it does not change Boost.SML structure, public APIs, acceptance surfaces, or the
effective operand contract already shipped through the generator -> graph -> processor -> kernel
chain.

</domain>

<decisions>
## Implementation Decisions

### Kernel Seam
- Keep the cutover inside `src/emel/kernel/aarch64/actions.hpp`, specifically the quantized
  branch of `execute_neon_mul_mat(...)`; do not add transition rows, events, or new runtime
  wrappers in this phase.
- Preserve the existing request validation, row-byte layout, and per-column `q8_K` staging that
  `execute_neon_mul_mat(...)` already performs for quantized matmul.
- Replace only the maintained `q3_K x q8_K` row-dot execution path in Phase 18; `q6_K` remains on
  its current helper until Phase 19.
- Keep unsupported or non-AArch64 execution explicit and deterministic through the existing shared
  fallback behavior instead of widening the supported optimized contract.

### Operand And Performance Contract
- Reuse the existing `block_q8_k` operand pipeline and the AArch64 q3 block arithmetic already
  present in backend-local helpers; do not introduce dequantize-to-f32 fallbacks or new heap
  allocation.
- Preserve zero-allocation dispatch and the current bounded scratch usage already established in
  the backend hot path.
- Keep all runtime control flow unchanged and model this phase strictly as a data-plane helper
  replacement, with loops used only for bounded numeric iteration.
- Treat the shipped canonical operand class as fixed; correctness and performance claims must be
  about that exact maintained path, not a simplified substitute.

### Proof Surface
- Add failing-first proof at the maintained kernel seam before landing the vectorized q3 cutover.
- Keep Phase 18 proof on maintained kernel-facing surfaces such as
  `tests/kernel/aarch64_tests.cpp` and `tests/kernel/lifecycle_tests.cpp`; do not widen parity or
  benchmark publication in this phase.
- Make supported vectorized `q3_K` execution distinguishable from the prior scalar helper on the
  canonical operand path.
- Defer runtime-chain publication, cross-dtype coverage, and benchmark attribution to Phases 20
  and 21.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/kernel/aarch64/actions.hpp` already owns the AArch64 quantized matmul seam and already
  contains backend-local `dot_q3_k_q8_k_block_neon(...)` arithmetic.
- Phase 17 already established the maintained seam pattern for q2 via backend-local row helper
  wiring and backend-local path attribution.
- `tests/kernel/aarch64_tests.cpp`, `tests/kernel/lifecycle_tests.cpp`, and
  `tests/kernel/test_helpers.hpp` already exercise quantized AArch64 dispatch and now contain the
  q2 seam-proof pattern that Phase 18 can mirror for q3.

### Established Patterns
- Backend specialization lives in backend-local helpers and wrapper accessors while SML structure
  stays unchanged.
- Unsupported optimized cases remain explicit and deterministic rather than silently claiming
  optimized execution.
- Hot-path numeric work stays allocation-free and bounded inside action/detail helpers.

### Integration Points
- `src/emel/kernel/aarch64/actions.hpp` is the exact seam where quantized matmul currently
  re-quantizes `src1` into `q8_K` blocks and still dispatches `q3_K` through the shared scalar row
  helper.
- `src/emel/kernel/detail.hpp` remains the scalar parity reference for q3 row results and operand
  layout.
- `tests/kernel/aarch64_tests.cpp` is the narrowest maintained proof surface for direct kernel
  equivalence and path-selection evidence in this phase.

</code_context>

<specifics>
## Specific Ideas

- Introduce a backend-local q3 row helper that accumulates over the staged `block_q8_k` array via
  `dot_q3_k_q8_k_block_neon(...)` and mirrors the scalar row contract exactly.
- Add q3-focused tests that fail first until the maintained AArch64 path can distinguish
  vectorized q3 execution from the scalar helper on the canonical operand layout.
- Keep the `q6_K` branch untouched except where shared helper refactoring is needed to avoid
  duplication.

</specifics>

<deferred>
## Deferred Ideas

- Vectorized `q6_K` cutover and zero-allocation hot-path closure belong to Phase 19.
- Runtime-chain proof, parity publication, and benchmark attribution belong to Phases 20 and 21.

</deferred>

---
*Phase: 18-vectorized-q3-k-kernel*
*Context gathered: 2026-03-22*
