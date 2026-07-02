# Phase 240: x86_64 Flash Attention AVX2/FMA Kernel - Context

**Gathered:** 2026-06-25
**Status:** Ready for planning
**Mode:** Auto-generated (autonomous processor-support phase)

<domain>
## Phase Boundary

Port the maintained flash-attention optimization pattern to an EMEL-owned
x86_64 AVX2/FMA implementation for this Ryzen AVX2/FMA/F16C host. This phase
must route supported flash-attention requests through an optimized x86_64
kernel, preserve persistent workspace reuse, and keep unsupported requests on
explicit shared or invalid paths. It must not implement quantized q2_K/q3_K/q6_K
matmul kernels, runtime parity publication, or benchmark attribution; those are
active Phase 241-244 obligations.

</domain>

<decisions>
## Implementation Decisions

### Kernel Contract
- Implement a native x86_64 flash-attention helper in `src/emel/kernel/x86_64`
  using AVX2/FMA numeric work and F16C conversions for f16 K/V operand handling.
- Match the AArch64 one-chunk f16 K/V operand class: f32 Q is rounded into f16,
  K/V are consumed as f16, accumulation uses the existing f16 workspace buffer,
  and output is converted back to f32.
- Do not claim native FP16 arithmetic. F16C is a conversion capability; all x86
  vector arithmetic remains f32 AVX2/FMA.
- Keep all optimized kernel code in `src/`; parity and benchmark tools may only
  observe through public dispatch surfaces.

### Routing Contract
- Put supported optimized-path selection in `x86_64/guards.hpp` and
  `x86_64/sm.hpp`, analogous to the AArch64 flash route.
- Add explicit x86_64 optimized/shared flash counters so tests and active parity
  attribution can distinguish optimized execution from shared fallback.
- Keep unsupported shapes, feature contracts, and workspace constraints on the
  existing shared or invalid paths; do not silently label them optimized.

### Verification Contract
- Add failing-first x86_64 tests before source changes: optimized dispatch
  counter, shared fallback counter, persistent workspace reuse, and numeric
  comparison against the maintained shared/reference helper.
- Verify through public actor dispatch or route-owned detail functions, not by
  tool-only scaffolds.
- Do not update `snapshots/bench/benchmarks.txt` in this phase unless explicit
  snapshot approval is provided.

### the agent's Discretion
- Prefer the smallest x86_64 implementation that proves AVX2/FMA/F16C flash
  support and keeps active follow-on runtime/benchmark attribution
  straightforward.
- Reuse the existing `flash_attn_workspace` instead of adding per-dispatch
  allocation or new transient context fields.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/kernel/aarch64/actions.hpp` contains the optimized NEON flash
  precedent: `run_flash_attn_ext_f16kv_one_chunk_neon_unchecked`,
  `can_run_neon_flash_attn_ext_f16kv_one_chunk_request`, and
  `exec_simd_flash_attn_ext_f16kv_one_chunk`.
- `src/emel/kernel/aarch64/guards.hpp` and `sm.hpp` route optimized flash before
  the shared flash path, then invalid fallback.
- `src/emel/kernel/detail.hpp` owns the shared flash workspace, f16 conversion
  helpers, active-token handling, and scalar workspace fallback.
- `tests/kernel/test_helpers.hpp` provides flash fixtures and reference helpers.

### Established Patterns
- x86_64 SIMD helpers live in `src/emel/kernel/x86_64/actions.hpp`, with feature
  predicates in guards and transition rows in `sm.hpp`.
- The current x86_64 flash route accepts `op_flash_attn_ext` through
  `guard::valid_op_flash_attn_ext` and `action::exec_op_flash_attn_ext`, which
  calls the shared workspace helper.
- Phase 239 added x86_64 AVX2/FMA/F16C feature-contract fields and actor
  accessors that this phase can use for optimized flash eligibility.

### Integration Points
- Add x86 flash helpers and action aliases in `src/emel/kernel/x86_64/actions.hpp`.
- Add optimized/shared flash guards in `src/emel/kernel/x86_64/guards.hpp`.
- Add optimized/shared flash route counters in
  `src/emel/kernel/x86_64/context.hpp` and public accessors in `sm.hpp`.
- Add optimized flash transition rows before the shared flash row in
  `src/emel/kernel/x86_64/sm.hpp`.
- Add focused tests in `tests/kernel/x86_64_tests.cpp`.

</code_context>

<specifics>
## Specific Ideas

- The current host contract is Ryzen 9 5950X with AVX2, FMA, and F16C available.
- The optimized flash route should require all three: AVX2 for vector lanes, FMA
  for fused f32 accumulation, and F16C for f16 K/V conversion.
- The test surface should prove optimized and shared counters separately so
  Phase 243/244 attribution can build on source-backed evidence.

</specifics>

<active_follow_on_scope>
## Active Follow-On Scope

- Phase 241: AVX2/FMA q2_K/q3_K kernels.
- Phase 242: AVX2/FMA q6_K and hot-path operand-fidelity proof.
- Phase 243: maintained runtime integration and parity proof.
- Phase 244: benchmark attribution and publication truth.

</active_follow_on_scope>

---

*Phase: 240-x86-64-flash-attention-avx2-fma-kernel*
*Context gathered: 2026-06-25*
