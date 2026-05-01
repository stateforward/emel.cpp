# Phase 146: Text Generator Explicit Compute Outcome Modeling Closure - Context

**Gathered:** 2026-04-30
**Status:** Ready for planning
**Mode:** Autonomous, source-backed gap closure

<domain>
## Phase Boundary

Close the remaining v1.17 milestone audit blockers for `TEXTGEN-04` and `TEXTGEN-07`.
The source-backed problem is not missing documentation: graph compute request preconditions and
some route readiness checks still live in action-called `text/generator/detail.hpp` run-kernel
wrappers. Those helpers decide invalid/backend outcomes through runtime branching, `err_out`, and
callback return values after the parent/prefill actors have already selected a route.

This phase owns the maintained text generator path only. It may update model, snapshot, and
benchmark evidence when needed to prove the maintained path, but evidence cannot substitute for
explicit behavior modeling.

</domain>

<decisions>
## Implementation Decisions

### Explicit Behavior Modeling

All compute request preconditions that select acceptance, invalid request, backend error, missing
output, wrong step kind, chunk readiness, or route availability must be modeled as guard-owned
predicates and destination-first SML transition rows in the generator and prefill actors.

### Action/Detail Boundary

Action-called run-kernel callbacks in `detail.hpp` must only execute the already-selected numeric
kernel path. They must not branch on runtime request shape, step kind, backend readiness, selected
output pointers, chunk readiness, or callback error channel selection.

### Evidence

Tests must inspect live source and maintained public actor paths. Artifact-only claims from roadmap,
summary, verification, or validation files are insufficient for closeout.

</decisions>

<code_context>
## Existing Code Insights

- Parent decode route selection already has explicit SML decision states in
  `src/emel/text/generator/sm.hpp`, but the selected run-kernel callbacks still repeat validation.
- Prefill route selection is explicit in `src/emel/text/generator/prefill/sm.hpp`, but chunk and
  preselected run-kernel wrappers still reject missing route prerequisites inside `detail.hpp`.
- `action::request_phase_compute<...>` and
  `action::request_phase_compute_preselected_argmax<...>` bind the callback functions into the
  graph processor.
- Existing regression tests in `tests/text/generator/lifecycle_tests.cpp` scan source for prior
  route-selection regressions and should be extended to cover Phase 146 failure modes.

</code_context>

<specifics>
## Specific Ideas

- Add guard predicates for graph compute request readiness before compute dispatch.
- Add invalid/backend transitions from compute route decision states before any action dispatch.
- Simplify the audited run-kernel wrappers to dereference the prevalidated request and run the
  selected kernel without `err_out` validation/outcome branching.
- Replace direct detail-wrapper malformed-request tests with actor/source regressions that prove the
  rejection path is guard-modeled.

</specifics>

<deferred>
## Deferred Ideas

None.

</deferred>
