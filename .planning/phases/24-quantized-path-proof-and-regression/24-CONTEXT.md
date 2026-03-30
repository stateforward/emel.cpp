# Phase 24: Quantized Path Proof And Regression - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 24 must turn the published Phase 23 runtime contract into maintained proof surfaces that
fail if the canonical ARM generation path regresses away from the approved quantized contract.
This phase is about paritychecker/runtime proof and regression hardening, not about changing the
data-plane contract again.

This phase stays inside the existing generator -> graph -> processor -> kernel chain and the
maintained `tools/paritychecker` plus test acceptance surfaces. It must not change Boost.SML actor
structure, public C API boundaries, or benchmark/docs publication scope without explicit approval.

</domain>

<decisions>
## Implementation Decisions

### Locked From Phase 23
- The shipped generator wrapper now exposes additive runtime contract counts:
  `native_quantized`, `approved_dense_f32_by_contract`, `disallowed_fallback`, and
  `explicit_no_claim`.
- The supported canonical initialized runtime currently proves the truthful `8/4/0/0` contract.
- Unsupported quantized-stage families still belong to explicit `no-claim` proof, not to a fake
  supported fallback path.

### Phase 24 Question
- Promote the Phase 23 runtime contract into maintained paritychecker and regression checks so the
  canonical `1/10/100/1000` workload fails deterministically if `disallowed_fallback` or
  `explicit_no_claim` ever appears on the supported path.
- Keep approved dense-f32-by-contract seams visible instead of collapsing everything into a
  misleading “fully quantized” claim.

### Guardrails
- Do not change Boost.SML transition tables or actor ownership without explicit user approval.
- Keep Phase 24 focused on maintained proof and regression surfaces; benchmark/docs refresh stays
  in Phase 25.
- Prefer the shipped generator runtime contract as the proof source, using model-audit
  recomputation only as a consistency check if needed.

</decisions>

<canonical_refs>
## Canonical References

- `.planning/ROADMAP.md` — Phase 24 goal, requirements, and success criteria.
- `.planning/REQUIREMENTS.md` — `ATTR-01`, `VER-04`, and `PAR-05`.
- `.planning/PROJECT.md` — validated Phase 23 zero-gap decision and remaining milestone focus.
- `.planning/phases/23-arm-quantized-path-closure/23-VERIFICATION.md` — current `8/4/0/0`
  runtime contract truth.
- `AGENTS.md` and `docs/rules/sml.rules.md` — no machine-structure change without approval and RTC
  actor constraints.

</canonical_refs>

<code_context>
## Existing Code Insights

- `src/emel/generator/sm.hpp` now exposes the shipped runtime contract counts directly on the
  generator wrapper.
- `tools/paritychecker/parity_runner.cpp` still recomputes the quantized-path audit from
  `model_data` and only prints it; it does not yet fail from the shipped runtime contract if the
  canonical path regresses.
- `tools/paritychecker/paritychecker_tests.cpp` currently checks for the presence of audit strings
  and dispatch metrics across `1/10/100/1000`, but it does not yet assert the exact `8/4/0/0`
  contract or regression-failure semantics.
- `tests/generator/lifecycle_tests.cpp` proves the initialized supported fixture reports
  `8/4/0/0`, and Phase 22 already covers unsupported-stage explicit `no-claim` at the audit level.

</code_context>

<specifics>
## Specific Ideas

- Make `tools/paritychecker --generation` consume the shipped runtime contract counts, publish them
  explicitly, and fail if the canonical supported path reports any `disallowed_fallback` or
  `explicit_no_claim` stages.
- Extend maintained paritychecker tests so all decode lengths `1`, `10`, `100`, and `1000`
  assert the exact supported `8/4/0/0` contract instead of only string presence.
- Add the narrowest regression coverage needed so supported-path proof and unsupported no-claim
  proof stay distinct and deterministic.

</specifics>

<deferred>
## Deferred Ideas

- Benchmark/docs attribution refresh remains Phase 25.
- Any broader model-matrix or non-canonical proof expansion remains outside this milestone.

</deferred>

---
*Phase: 24-quantized-path-proof-and-regression*
*Context gathered: 2026-03-25*
