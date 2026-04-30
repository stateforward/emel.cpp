# Phase 140: Text Generator Boundary Contract Closure - Context

**Gathered:** 2026-04-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Close the source-backed v1.17 boundary gaps for `TEXTGEN-01` and `TEXTGEN-06` without changing
generation runtime behavior. This phase decides and documents the text-generator include-root truth
and makes maintained boundary checks reject hidden parity/benchmark actor-internal bridges.

</domain>

<decisions>
## Implementation Decisions

### Boundary Contract
- Treat `emel/text/generator/**` as the include path served by the repo's existing `src` include
  root, consistent with current `emel_core` CMake wiring.
- Do not add duplicate forwarding headers unless implementation evidence shows a real installed
  header boundary is required.
- Preserve `src/emel/text/generator/**` as the single canonical implementation root.

### Boundary Enforcement
- Move parity/benchmark actor-internal bridge checks into `scripts/check_domain_boundaries.sh`.
- Scope the new check to maintained generation parity/benchmark entrypoints so tests may still
  contain regression strings.
- Keep the check focused on `detail.hpp`, `actions.hpp`, `guards.hpp`, actor-internal namespaces,
  prefill guard internals, and the deleted diagnostic bridge name.

### the agent's Discretion
All implementation choices may follow the smallest source-backed change that closes the audit gap
without widening model, sampling, or performance scope.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/check_domain_boundaries.sh` already centralizes stale domain/root leak checks.
- `tools/bench/bench_runner_tests.cpp` and paritychecker tests already contain focused bridge
  regression checks that can guide the maintained script pattern.

### Established Patterns
- `emel_core` publishes both `include` and `src` as include roots.
- Existing component headers are included by logical paths such as `emel/text/generator/sm.hpp`.

### Integration Points
- Maintained generation benchmark path: `tools/bench/generation_bench.cpp`.
- Maintained generation parity path: `tools/paritychecker/parity_runner.cpp` and public runner
  declarations in `tools/paritychecker/parity_runner.hpp`.

</code_context>

<specifics>
## Specific Ideas

No runtime behavior changes. This phase is boundary evidence and enforcement only.

</specifics>

<deferred>
## Deferred Ideas

Physical installed header packaging remains out of scope unless a future public packaging milestone
requires it.

</deferred>
