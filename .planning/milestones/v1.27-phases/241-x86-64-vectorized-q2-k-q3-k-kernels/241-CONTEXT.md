# Phase 241: x86_64 Vectorized q2_K/q3_K Kernels - Context

**Gathered:** 2026-06-25
**Status:** Ready for planning
**Mode:** Auto-generated (autonomous processor-support phase)

<domain>
## Phase Boundary

Land EMEL-owned x86_64 AVX2/FMA `q2_K x q8_K` and `q3_K x q8_K` kernels for
the maintained quantized `op_mul_mat` hot path. This phase must prove optimized
execution and shared fallback/no-claim behavior for q2_K and q3_K through the
kernel actor route. It must not implement q6_K, runtime generator integration,
or benchmark publication; those remain active Phase 242-244 obligations.

</domain>

<decisions>
## Implementation Decisions

### Kernel Contract
- Implement native x86_64 row kernels in `src/emel/kernel/x86_64/actions.hpp`
  for `q2_K x q8_K` and `q3_K x q8_K`.
- Preserve the same effective operand class as the maintained scalar/AArch64
  paths: block-q2_K or block-q3_K LHS and block-q8_K RHS, no whole-tensor
  dequantize-to-f32 substitution in the hot path.
- Use AVX2/FMA for vectorized accumulation where it improves the native row
  path, with scalar tail handling only inside the already-selected kernel.
- Keep Phase 241 limited to q2_K and q3_K; q6_K remains Phase 242.

### Routing Contract
- Add x86_64 guards/transitions for supported q2_K/q3_K `op_mul_mat` before the
  generic f32 AVX2 and shared scalar routes.
- Add optimized/shared q2/q3 route counters and actor accessors analogous to the
  AArch64 route counters.
- Keep runtime behavior choice in `guards.hpp` and `sm.hpp`; action/detail code
  must execute only an already-selected q2 or q3 path.

### Verification Contract
- Add failing-first tests proving q2_K and q3_K optimized counters are missing or
  shared-only before implementation.
- Add correctness tests comparing x86 optimized row/mul_mat output to the
  maintained scalar/reference oracle for representative blocks, multiple block
  groups, tails, and accumulation behavior.
- Prove disabled feature contracts take the shared q2/q3 route without claiming
  optimized execution.

### the agent's Discretion
- Start with row-level kernels plus actor-route tests if that is the smallest
  source-backed route to `XQK-01`/`XQK-02`.
- Reuse test helpers from AArch64 quantized tests when possible instead of
  inventing parallel fixtures.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/kernel/detail.hpp` owns scalar q2_K/q3_K/q8_K block structures and
  scalar dot helpers.
- `src/emel/kernel/aarch64/actions.hpp` contains NEON q2/q3 row kernels and
  `op_mul_mat` route/counter precedent.
- `tests/kernel/aarch64_tests.cpp` has row-level q2/q3 correctness fixtures and
  actor-route tests for optimized/shared q2/q3 dispatch.
- `tests/kernel/test_helpers.hpp` has q8_K vector source helpers and quantized
  tensor-view construction helpers.

### Established Patterns
- Quantized optimized routes increment optimized q-format counters; shared
  scalar routes increment shared q-format counters.
- The actor surface exposes route counters through `sm.hpp` and `kernel/any.hpp`
  consumes them when available.
- Fallback/no-claim behavior is tested by disabling host SIMD support in the
  machine context and proving shared counters increment.

### Integration Points
- `src/emel/kernel/x86_64/actions.hpp`: row kernels, selected-route actions, and
  shared/optimized counter increments.
- `src/emel/kernel/x86_64/guards.hpp`: q2/q3 optimized route predicates and
  generic q2/q3 exclusion from shared route predicates.
- `src/emel/kernel/x86_64/context.hpp`: q2/q3 route counters.
- `src/emel/kernel/x86_64/sm.hpp`: q2/q3 transition rows and actor accessors.
- `tests/kernel/x86_64_tests.cpp`: focused row and actor-route tests.

</code_context>

<specifics>
## Specific Ideas

- The Phase 241 request is exactly to bring this Ryzen AVX2/FMA processor toward
  the same support standard as the NEON path, so q2_K/q3_K must be native EMEL
  kernels, not benchmark-only or whole-tensor f32 fallback code.
- The x86 feature contract from Phase 239 provides AVX2/FMA/F16C support booleans
  that can gate q2/q3 optimized routes. q2/q3 integer unpacking itself should
  not claim AVX-512, VNNI, AMX, BF16, or native FP16 support.

</specifics>

<active_follow_on_scope>
## Active Follow-On Scope

- Phase 242: AVX2/FMA q6_K and hot-path operand-fidelity proof.
- Phase 243: maintained runtime integration and parity proof.
- Phase 244: benchmark attribution and publication truth.

</active_follow_on_scope>

---

*Phase: 241-x86-64-vectorized-q2-k-q3-k-kernels*
*Context gathered: 2026-06-25*
