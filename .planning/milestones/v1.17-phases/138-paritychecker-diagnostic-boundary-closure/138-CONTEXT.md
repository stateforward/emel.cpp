# Phase 138: Paritychecker Diagnostic Boundary Closure - Context

**Gathered:** 2026-04-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Close the v1.17 audit blocker where paritychecker still reaches text-generator actor internals
through `tools/paritychecker/generation_internal_diagnostics.hpp`, despite the runner no longer
including `emel/text/generator/detail.hpp` directly.

</domain>

<decisions>
## Implementation Decisions

### Boundary Closure
- Remove the paritychecker diagnostic bridge rather than replacing it with another tool-owned
  wrapper over generator actor internals.
- Keep the maintained generation parity result path actor-driven through public
  `emel::text::generator::sm` initialize and generate events.
- Do not widen model family, fixture, sampling, or performance scope while closing this gap.
- If optional diagnostic attribution cannot be produced through a public/non-actor boundary in
  this phase, remove or reject that optional diagnostic path instead of preserving a hidden
  `detail.hpp` bypass.

### Verification
- Add source regression coverage that scans both `parity_runner.cpp` and paritychecker bridge
  headers for hidden generator actor-internal includes or names.
- Preserve existing maintained generation parity, benchmark, and domain-boundary gates.
- Treat source-backed audit contradiction as authoritative over Phase 137 artifacts.

### the agent's Discretion
Implementation details are at the agent's discretion. Prefer the smallest code change that removes
the live actor-internal dependency and keeps maintained generation proof truthful.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/paritychecker/parity_runner.cpp` already has a public actor-driven generation path through
  `run_emel_initialize_generator(...)`, `run_emel_generate(...)`, and
  `state.generator->process_event(...)`.
- `tools/paritychecker/paritychecker_tests.cpp` already contains a source-level regression for
  direct `parity_runner.cpp` actor-internal references.
- `scripts/check_domain_boundaries.sh` already checks stale generator ownership names.

### Established Patterns
- Milestone closeout claims must be source-backed, not artifact-backed.
- Parity and benchmark harnesses must not reach into actor `actions.hpp`, `detail.hpp`,
  `detail.cpp`, or guard helpers directly.
- Optional diagnostic output must not be presented as maintained proof when it bypasses actor
  boundaries.

### Integration Points
- `tools/paritychecker/generation_internal_diagnostics.hpp`
- `tools/paritychecker/parity_runner.cpp`
- `tools/paritychecker/paritychecker_tests.cpp`
- `.planning/v1.17-MILESTONE-AUDIT.md`

</code_context>

<specifics>
## Specific Ideas

Remove or replace `generation_internal_diagnostics.hpp` so no paritychecker source includes or
re-exports `emel/text/generator/detail.hpp` or `emel::text::generator::detail::*`.

</specifics>

<deferred>
## Deferred Ideas

Future richer generation attribution can be designed as a source-owned public event or metrics
contract if it is still needed after v1.17 closeout.

</deferred>
