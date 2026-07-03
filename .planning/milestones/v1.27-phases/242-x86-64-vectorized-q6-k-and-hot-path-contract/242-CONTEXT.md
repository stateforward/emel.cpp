# Phase 242: x86_64 Vectorized q6_K and Hot-Path Contract - Context

**Gathered:** 2026-06-25
**Status:** Ready for planning
**Mode:** Auto-generated (autonomous processor-support phase)

<domain>
## Phase Boundary

Add the EMEL-owned x86_64 AVX2/FMA `q6_K x q8_K` kernel for the maintained
quantized `op_mul_mat` hot path and prove the q2_K/q3_K/q6_K optimized x86_64
routes preserve the hot-path contract: same effective operand class, no
whole-tensor dequantize-to-f32 substitution, and no dispatch-time allocation.
This phase does not own runtime generator integration, parity publication, or
benchmark publication; those remain active Phase 243-244 obligations.

</domain>

<decisions>
## Implementation Decisions

### Kernel Contract
- Implement a native x86_64 `q6_K x q8_K` row kernel in
  `src/emel/kernel/x86_64/actions.hpp`.
- Preserve the same operand path as scalar/AArch64 q6: block-q6_K LHS,
  block-q8_K RHS produced by the maintained q8_K quantizer, and f32 output.
- Do not add whole-tensor dequantize-to-f32 hot-path substitution.
- Keep q6_K scope to the maintained unpacked block-q6_K route; packed/prepared
  q6 vector variants are not required for this phase unless already needed by
  the maintained x86 route under test.

### Routing Contract
- Add x86_64 q6 optimized route predicates in `guards.hpp`, selected-route
  actions in `actions.hpp`, counters in `context.hpp`, actor accessors in
  `sm.hpp`, and transition rows before generic f32 SIMD/shared scalar
  `op_mul_mat` routes.
- Keep runtime behavior choice in guards and SML transitions; q6 actions must
  execute only the already-selected q6 path.
- Add shared q6 counter attribution for the scalar fallback/no-claim path.

### Hot-Path Contract Proof
- Prove q2_K/q3_K/q6_K optimized routes consume q*_K blocks and q8_K RHS blocks,
  not whole-tensor dequantized f32 intermediates.
- Prove supported optimized dispatch performs no heap allocation by exercising
  q2/q3/q6 route calls under the repo's maintained allocation/checking pattern
  or a focused deterministic allocation guard.
- Keep validation source-backed: tests must drive public actor dispatch, not
  private action helpers directly.

### the agent's Discretion
- Start with row helper correctness plus actor route tests for q6_K, then add
  the narrowest allocation/operand-fidelity assertions that are source-backed
  and maintainable.
- Reuse Phase 241 x86 fixtures and AArch64 q6 test patterns where practical.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/kernel/detail.hpp` owns scalar `dot_q6_k_q8_k_block_scalar` and
  `dot_q6_k_q8_k_row_scalar`.
- `src/emel/kernel/aarch64/actions.hpp` contains NEON q6 row and `op_mul_mat`
  route/counter precedent.
- `tests/kernel/aarch64_tests.cpp` has q6 row correctness and actor route
  examples.
- Phase 241 added x86 q2/q3 helpers, route counters, guards/transitions, and
  focused x86 tests that can be extended for q6.

### Established Patterns
- Quantized optimized routes increment optimized q-format counters; shared
  scalar routes increment shared q-format counters.
- Feature-disabled machine contexts prove fallback/no-claim behavior.
- Focused x86 tests compare optimized row/mul_mat output against scalar q*_K x
  q8_K oracles and then validate actor route attribution.

### Integration Points
- `src/emel/kernel/x86_64/actions.hpp`: q6 row kernel, selected-route action,
  shared/optimized counter increments, and hot-path helper code.
- `src/emel/kernel/x86_64/guards.hpp`: q6 optimized route predicate and generic
  q6 exclusion from shared route predicates.
- `src/emel/kernel/x86_64/context.hpp`: q6 route counters.
- `src/emel/kernel/x86_64/sm.hpp`: q6 transition row and actor accessors.
- `tests/kernel/x86_64_tests.cpp`: q6 row/route tests and hot-path contract
  tests.

</code_context>

<specifics>
## Specific Ideas

- Phase 242 closes the quantized-kernel set named in v1.27: q2_K, q3_K, and
  q6_K on this AVX2/FMA host.
- The hot-path contract is an implementation obligation in this phase, not a
  publication-only claim: supported optimized requests must stay block-native
  and allocation-free during dispatch.

</specifics>

<active_follow_on_scope>
## Active Follow-On Scope

- Phase 243: maintained runtime integration and parity proof.
- Phase 244: benchmark attribution and publication truth.

</active_follow_on_scope>

---

*Phase: 242-x86-64-vectorized-q6-k-and-hot-path-contract*
*Context gathered: 2026-06-25*
