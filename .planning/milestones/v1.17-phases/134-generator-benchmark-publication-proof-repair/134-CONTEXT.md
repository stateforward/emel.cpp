# Phase 134: Generator Benchmark Publication Proof Repair - Context

**Gathered:** 2026-04-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Close the v1.17 audit blocker where maintained generation benchmark publication probes derive
EMEL stage metadata by constructing generator internals and calling `emel::text::generator`
`detail`, `action`, and guard helpers directly. The maintained EMEL publication proof must be
actor-driven through public generator events or use non-actor-owned benchmark/reference surfaces
that do not bypass `emel::text::generator::sm`.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- This is a pure benchmark/proof-surface repair phase.
- Prefer removing unsupported actor-internal stage attribution over inventing new public generator
  APIs.
- Preserve existing benchmark output shape when possible so downstream tooling does not need a
  schema migration.
- Do not introduce a new model family, fixture, sampling policy, or performance claim.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `run_emel_generate(...)` in `tools/bench/generation_bench.cpp` already drives
  `emel::text::generator::sm` through public `event::generate`.
- `tools/bench/bench_main.cpp` already prints generation stage probe fields; zero-valued fields
  can preserve output shape while avoiding false actor-internal attribution.
- `tools/bench/bench_runner_tests.cpp` verifies that the stage probe line and fields are present.

### Established Patterns
- EMEL benchmark lanes must stay lane-isolated from reference objects.
- Parity and benchmark harnesses must not reach into actor `actions.hpp`, `detail.hpp`,
  `detail.cpp`, or guard helpers directly for actor behavior proof.
- Source-backed closeout proof should favor truthful metadata over a fabricated or bypassed stage
  breakdown.

### Integration Points
- `capture_generation_stage_probe(...)` is called for the current publication fixture in the
  generation compare lane.
- Reference-side stage probes may continue using reference-side APIs because they are isolated to
  the reference lane.

</code_context>

<specifics>
## Specific Ideas

Keep the stage probe line, but mark the EMEL prefill contract as an actor-owned public-generate
measurement and leave unsupported actor-internal stage attribution fields at zero.

</specifics>

<deferred>
## Deferred Ideas

Adding a first-class public generator telemetry event is deferred to a future phase.

</deferred>
