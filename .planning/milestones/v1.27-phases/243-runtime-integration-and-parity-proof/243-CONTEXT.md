# Phase 243: Runtime Integration and Parity Proof - Context

**Gathered:** 2026-06-25
**Status:** Ready for planning
**Mode:** Auto-generated (autonomous processor-support phase)

<domain>
## Phase Boundary

Adopt the x86_64 AVX2/FMA kernel work from Phases 239-242 in the maintained
generator -> graph -> processor -> kernel proof surfaces. This phase does not
add new numeric kernels; it proves the shipped runtime chain selects the new
x86_64 optimized routes where the maintained generation fixture actually uses
q2_K/q3_K/q6_K tensors, and that paritychecker publishes the corresponding
attribution for `1`, `10`, `100`, and `1000` token generation runs.

</domain>

<decisions>
## Implementation Decisions

### Runtime Diagnostics Contract
- Use the existing generator `capture_diagnostics` event as the runtime proof
  surface. It already exposes `kernel_kind`, total kernel dispatch count,
  flash attribution, q2/q3/q6 optimized and shared counters, and quantized
  contract stage counts.
- Do not add public API or C ABI surface. Phase 243 proof stays inside
  maintained tests and paritychecker attribution.
- Do not reach into actor actions or private helpers from tests or tools; drive
  generator proof through `process_event(...)` and public generator events.

### Maintained Generation Fixture
- Use `generator_fixture::model_variant::quantized_contract` in
  `tests/text/generator/lifecycle_tests.cpp` for source-backed generator-chain
  assertions. Its tensor setup assigns q2_K/q3_K/q6_K to maintained model
  stages.
- On x86_64 hosts, require optimized q2/q3/q6 dispatch counters to be positive
  and shared q2/q3/q6 counters to stay zero for the quantized-contract generate
  path.
- On non-x86 hosts, keep existing platform-specific expectations intact.

### Paritychecker Proof
- Update `tools/paritychecker/parity_engines.cpp` so generation parity accepts
  and requires x86_64 optimized q2/q3/q6 attribution when the maintained
  generation fixture runs on the x86_64 kernel kind.
- Keep non-x86 and AArch64 expectations explicit; do not claim x86 optimized
  attribution on other kernel kinds.
- Extend existing `tools/paritychecker/paritychecker_tests.cpp` attribution
  checks so emitted `quantized_dispatch:` output proves the x86 counters are
  positive and shared counters are zero.

### Validation
- Run focused generator quantized-contract doctest cases directly when the
  broad `emel_tests_generator_and_runtime` shard is blocked by unrelated dirty
  embedding fixture failures.
- Run paritychecker generation proof for the maintained fixture and token
  counts `1`, `10`, `100`, and `1000` when fixture assets are present.
- Keep benchmark snapshot approval as the shared milestone closeout gate; do
  not update snapshots without explicit approval.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/text/generator/actions.hpp` fills generator diagnostics from
  `ctx.compute.backend.kernel.*_dispatch_count()` accessors.
- `src/emel/kernel/any.hpp` exposes q2/q3/q6 optimized/shared counters across
  x86_64 and AArch64 kernel actors.
- `tests/text/generator/lifecycle_tests.cpp` contains the
  `quantized_contract` generator fixture and existing quantized contract tests.
- `tools/paritychecker/parity_engines.cpp` prints `quantized_dispatch:` and
  validates runtime quantized attribution during generation parity.
- `tools/paritychecker/paritychecker_tests.cpp` already parses generation
  attribution output.

### Integration Points
- `tests/text/generator/lifecycle_tests.cpp`: strengthen maintained generator
  diagnostics assertions for x86_64 q2/q3/q6 optimized dispatch.
- `tools/paritychecker/parity_engines.cpp`: require the x86_64 generation
  parity path to show optimized q2/q3/q6 dispatch and zero shared q2/q3/q6
  dispatch.
- `tools/paritychecker/paritychecker_tests.cpp`: assert the emitted
  `quantized_dispatch:` metrics match the x86_64 runtime contract.

</code_context>

<specifics>
## Specific Ideas

- The generator-level proof must distinguish two facts:
  1. f32/default fixtures do not claim quantized optimized dispatch.
  2. the quantized-contract fixture does claim optimized q2/q3/q6 dispatch on
     x86_64 and does not fall back to shared q2/q3/q6 dispatch.
- Paritychecker should fail if x86_64 generation parity succeeds numerically
  while the optimized attribution counters are missing.

</specifics>

<active_next_scope>
## Active Next Scope

- Phase 244: benchmark attribution and publication truth after Phase 243
  runtime/parity proof is source-backed.

</active_next_scope>

---

*Phase: 243-runtime-integration-and-parity-proof*
*Context gathered: 2026-06-25*
