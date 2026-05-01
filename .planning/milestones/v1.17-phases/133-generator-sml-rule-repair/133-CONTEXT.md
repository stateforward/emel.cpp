# Phase 133: Generator SML Rule Repair - Context

**Gathered:** 2026-04-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Close the v1.17 audit blocker where `emel::text::generator::sm` still performs runtime behavior
selection in the public `process_event(event::initialize)` wrapper. This phase must model
initialize readiness and invalid-request outcomes through explicit Stateforward.SML guards, states, and
transitions while preserving generator initializer and prefill child ownership under
`src/emel/text/generator/**`.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- This is a pure infrastructure/SML-rule repair phase.
- Keep generation request/output behavior stable.
- Prefer the existing generator machine structure and naming conventions.
- Do not widen model, tokenizer, formatter, sampling, or benchmark scope.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/text/generator/sm.hpp` owns the parent generator transition table and public wrapper.
- `src/emel/text/generator/guards.hpp` is the correct home for runtime readiness predicates.
- `src/emel/text/generator/actions.hpp` already owns bounded initialize error effects and
  internal initialize dispatch actions.
- `tests/text/generator/**` contains lifecycle, action/guard, initializer, prefill, and detail
  tests for focused regression coverage.

### Established Patterns
- Runtime behavior choice belongs in `sm.hpp` transitions using guards from `guards.hpp`.
- Actions perform already-selected effects and must not choose behavior.
- Public callbacks are invoked synchronously inside the RTC chain and are not stored.

### Integration Points
- The repair must preserve `emel::Generator = emel::text::generator::sm` from
  `src/emel/machines.hpp`.
- The existing `generator_and_runtime` shard is the focused test target.

</code_context>

<specifics>
## Specific Ideas

No specific requirements beyond the audit finding and ROADMAP success criteria.

</specifics>

<deferred>
## Deferred Ideas

None.

</deferred>
