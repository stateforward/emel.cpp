# Phase 22: Quantized Path Audit And Contract - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 22 audits the maintained canonical ARM generation chain and makes the quantized operand-path
contract explicit. It must identify which shipped stages are native quantized, which dense-f32
stages remain approved by contract, and which branches would count as disallowed fallback or
explicit no-claim behavior.

This phase stays inside the existing canonical acceptance surfaces in `tools/paritychecker` and
`tools/bench`. It does not change Boost.SML actor structure, widen public APIs, or claim broader
model/backend support beyond the maintained ARM Llama-68M slice.

</domain>

<decisions>
## Implementation Decisions

### Audit Scope And Truth Source
- Audit the shipped generator -> graph -> processor -> kernel runtime chain, not a tool-only or
  synthetic kernel path.
- Ground every contract claim in EMEL-owned `src/` behavior first, then reflect that contract out
  through maintained proof surfaces.
- Keep the audit anchored to the canonical ARM `tests/models/Llama-68M-Chat-v1-Q2_K.gguf`
  workload already used by paritychecker and bench.

### Contract Classification
- Classify stages into exactly three buckets for this milestone: native quantized, approved
  dense-f32-by-contract, and disallowed fallback.
- Treat the existing quantized `op_mul_mat` path that repacks dense rhs activations into `q8_K`
  blocks as an approved dense-f32-by-contract stage unless Phase 22 evidence proves otherwise.
- Any unsupported or not-yet-ported quantized branch must become an explicit no-claim path rather
  than silently taking a misleading f32 or dequantize-to-f32 fallback.

### Guardrails
- Do not change Boost.SML transition tables or actor ownership in this phase without explicit user
  approval.
- Prefer additive observability and operator-inventory reporting over hidden fallback behavior or
  broad runtime rewrites.
- Keep proof and inventory surfaces deterministic, maintained, and aligned with current repo gates.

### the agent's Discretion
The agent can choose the exact artifact shape for the audit and inventory surfaces, provided the
output stays narrow, machine-readable enough for later proof phases, and clearly maps each stage to
one of the approved contract buckets above.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone Contract
- `.planning/ROADMAP.md` — Phase 22 goal, success criteria, plan slots, and current milestone
  guardrails.
- `.planning/REQUIREMENTS.md` — `AUD-01` and `PATH-02` definitions plus milestone traceability.
- `.planning/PROJECT.md` — milestone intent, validated history, and v1.5 scope boundaries.

### Architecture Rules
- `AGENTS.md` — repo-local engineering contract, especially no actor-structure changes without user
  approval and no misleading fallback claims.
- `docs/rules/sml.rules.md` — canonical Boost.SML semantics and RTC actor constraints.

### Prior Phase Decisions
- `.planning/milestones/v1.4-phases/19-vectorized-q6-k-kernel-and-hot-path-contract/19-CONTEXT.md`
  — prior quantized hot-path seam and operand-contract decisions.
- `.planning/milestones/v1.4-phases/20-runtime-integration-and-proof/20-CONTEXT.md` — runtime
  attribution pattern and canonical proof-surface decisions.
- `.planning/milestones/v1.4-phases/21-benchmark-attribution-and-impact/21-CONTEXT.md` — compare
  and benchmark publication constraints for quantized attribution.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/generator/sm.hpp` already exposes generation-level flash and quantized dispatch-count
  accessors for q2/q3/q6 runtime attribution.
- `src/emel/kernel/aarch64/context.hpp` already stores optimized/shared q2/q3/q6 counters plus
  flash counters at the backend seam.
- `src/emel/kernel/aarch64/actions.hpp` already implements the quantized `op_mul_mat` path that
  repacks dense rhs activations into bounded `q8_K` scratch before q2/q3/q6 row-dot execution.
- `tools/paritychecker/parity_runner.cpp` and `tools/bench/generation_bench.cpp` already exercise
  the canonical ARM Llama-68M generation slice and are the correct maintained proof surfaces.

### Established Patterns
- Runtime observability is additive and wrapper-local; prior phases exposed backend truth through
  accessors instead of actor rewrites.
- Canonical proof belongs on the maintained paritychecker and bench surfaces, while narrow kernel
  behavior is proven in targeted unit tests.
- Unsupported paths are supposed to publish truthful no-claim behavior instead of silently claiming
  optimized execution.

### Integration Points
- `src/emel/generator/detail.hpp` and `src/emel/generator/sm.hpp` are the narrowest shipped seams
  to trace generation-time backend usage.
- `src/emel/kernel/aarch64/actions.hpp`, `src/emel/kernel/aarch64/context.hpp`, and
  `src/emel/kernel/aarch64/sm.hpp` are the exact backend surfaces that define current quantized
  operand handling on ARM.
- `tests/kernel/lifecycle_tests.cpp`, `tests/kernel/aarch64_tests.cpp`,
  `tools/paritychecker/parity_runner.cpp`, and `tools/bench/generation_bench.cpp` are the current
  maintained evidence surfaces that later phases will extend.

</code_context>

<specifics>
## Specific Ideas

- Produce an operator-by-operator inventory for the canonical generation slice that explicitly
  labels whether each quantized stage is native quantized, dense-f32-by-contract, or disallowed
  fallback/no-claim.
- Make the audit explain the current dense-rhs-to-`q8_K` repack behavior clearly enough that
  Phase 23 can distinguish approved contract from closure work.
- Prefer proof hooks that later phases can reuse directly for regression, parity, and benchmark
  attribution instead of one-off documentation-only notes.

</specifics>

<deferred>
## Deferred Ideas

- Removing approved dense-f32-by-contract stages is Phase 23 only if the milestone decides those
  stages are truly disallowed for supported canonical requests.
- Full parity/regression enforcement remains Phase 24.
- Benchmark publication refresh remains Phase 25.

</deferred>

---
*Phase: 22-quantized-path-audit-and-contract*
*Context gathered: 2026-03-25*
