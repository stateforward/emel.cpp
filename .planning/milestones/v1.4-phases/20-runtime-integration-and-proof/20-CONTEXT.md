# Phase 20: Runtime Integration And Proof - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 20 adopts the complete q2/q3/q6 vectorized quantized kernel set in the shipped
generator -> graph -> processor -> kernel runtime chain and proves supported plus fallback
behavior without changing public APIs or actor structure.

This phase can widen proof surfaces from the backend seam into runtime-facing observability, but it
must still avoid actor rewrites, public API widening, queue-based orchestration, or tool-only
compute fallbacks.

</domain>

<decisions>
## Implementation Decisions

### Runtime Seam
- Keep runtime integration additive by exposing existing backend-local q2/q3/q6 attribution
  through `kernel::any` and `generator::sm`; do not change state-machine structure or public C
  API signatures.
- Reuse the generator’s existing backend observability pattern for flash dispatch counts as the
  model for quantized-path attribution.
- Keep canonical proof on the shipped quantized GGUF loading path rather than the tiny in-memory
  f32 generator fixture.

### Proof Surface
- Use `tools/paritychecker --generation` as the maintained runtime-chain proof for canonical
- quantized execution because it already loads the real Q2_K fixture through `generator::sm`.
- Extend maintained generator and parity tests to prove two cases:
  1. canonical quantized runtime requests claim vectorized q2/q3/q6 execution truthfully
  2. non-quantized or unsupported paths do not make false optimized quantized claims
- Treat `1/10` generation parity as the active gate for now per explicit user approval, while
  continuing to record deferred `100/1000` parity debt instead of hiding it.

### Guardrails
- Do not edit Boost.SML transition tables for generator, graph, processor, or kernel actors.
- Do not widen runtime wrappers beyond additive attribution accessors.
- Do not claim full `1/10/100/1000` parity closure in this phase while the user-approved longer
  decode defer remains active.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/kernel/any.hpp` already forwards flash optimized/shared attribution from concrete
  backends using `visit(...)` and `requires` checks.
- `src/emel/generator/sm.hpp` already exposes generation-time flash attribution via additive
  accessors that read from `kernel::any`.
- `tools/paritychecker/parity_runner.cpp` already prints kernel/flash runtime attribution and uses
  the shipped quantized generation path with `Llama-68M-Chat-v1-Q2_K.gguf`.
- `tests/generator/lifecycle_tests.cpp` already proves the f32 generator fixture reports flash
  attribution truthfully and can absorb additive quantized-path no-claim checks.

### Established Patterns
- Runtime observability should be additive and wrapper-local rather than altering actor flow.
- Canonical runtime proof belongs in `paritychecker`; toy fixtures should only prove negative/no-
  false-claim cases.
- User-approved verification debt can stay explicit in phase verification without blocking
  autonomous progress.

### Integration Points
- `src/emel/kernel/any.hpp` is the narrowest place to surface q2/q3/q6 optimized/shared counts
  from AArch64 without widening concrete backend APIs elsewhere.
- `src/emel/generator/sm.hpp` is the narrowest shipped runtime surface for generation attribution.
- `tools/paritychecker/parity_runner.cpp` and
  `tools/paritychecker/paritychecker_tests.cpp` are the maintained runtime publication surfaces for
  canonical quantized generation proof.

</code_context>

<specifics>
## Specific Ideas

- Add `optimized_q{2,3,6}_dispatch_count()` and `shared_q{2,3,6}_dispatch_count()` forwarding to
  `kernel::any`, then expose corresponding generation accessors from `generator::sm`.
- Extend paritychecker generation output with named q2/q3/q6 runtime attribution metrics and make
  the canonical AArch64 path require optimized hits with zero shared fallback claims.
- Narrow the enforced parity length gate to `1` and `10`, and document `100/1000` as deferred by
  explicit user decision rather than as hidden pass criteria.
- Add a generator lifecycle test that the f32 fixture reports zero quantized optimized/shared
  dispatch claims.

</specifics>

<deferred>
## Deferred Ideas

- Full `100/1000` generation parity closure remains deferred by user priority.
- Benchmark publication remains Phase 21.

</deferred>

---
*Phase: 20-runtime-integration-and-proof*
*Context gathered: 2026-03-22*
