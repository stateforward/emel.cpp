# Phase 159: Benchmark Runner Discovery And Registration - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Move benchmark runner suite metadata out of the shared orchestrator into a localized registration
surface. This phase should make runner lookup deterministic and auditable without changing build
target structure, adding manifests, or altering maintained benchmark behavior.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- Use a C++ registry source/header as the localized metadata surface for current in-process
  runners.
- Keep `bench_runner.cpp` as an orchestrator that consumes the registry rather than owning the
  broad static suite array.
- Keep tokenizer behavior equivalent by registering tokenizer as an ordinary suite entry.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `bench_cases.hpp` already defines the `test_case` append-function contract used by every
  benchmark family.
- Phase 158 added `runner_request` so orchestrator execution already has a normalized entrypoint.

### Established Patterns
- Registry-style benchmark variant discovery already exists for generation workloads and embedding
  variants.
- Focused source tests are used to prove ownership boundaries in tool code.

### Integration Points
- `bench_runner.cpp` should call registry functions for default and kernel suite lists.
- `bench_runner_tests.cpp` should check registration lookup and source ownership.

</code_context>

<specifics>
## Specific Ideas

No specific requirements beyond localizing runner registration and preserving deterministic lookup.

</specifics>

<deferred>
## Deferred Ideas

Independent build targets, dependency manifests, and quality-gate consumption remain deferred.

</deferred>
