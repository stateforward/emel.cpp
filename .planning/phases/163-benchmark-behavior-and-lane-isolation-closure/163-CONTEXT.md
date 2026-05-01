# Phase 163: Benchmark Behavior And Lane-Isolation Closure - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Close the milestone by proving maintained benchmark behavior and lane isolation still hold after
the orchestrator, runner, build, registry, manifest, and gate refactors. This phase should add
verification coverage, not new benchmark semantics.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- Add source checks for shared benchmark orchestration rather than changing runtime code.
- Reuse `bench_runner_tests.cpp` because it already exercises maintained generation and
  diarization output schemas.
- Treat shared benchmark files as lane-neutral: they must not include actor internals or own
  EMEL/reference runtime objects.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `bench_runner_tests.cpp` already runs generation JSONL, diarization JSONL, shim delegation,
  runner contract, registry, build-target, manifest, and source-boundary checks.
- Phase 159 moved broad suite registration out of the orchestrator.
- Phase 162 proved the benchmark manifest is consumed by the quality gate before benchmark skip
  decisions.

### Integration Points
- Shared files are `bench_main.cpp`, `bench_runner.*`, `bench_runner_contract.hpp`,
  `bench_runner_registry.*`, and `bench_dependency_manifest.*`.
- Maintained behavior evidence comes from the full-suite `bench_runner_tests` target.

</code_context>

<specifics>
## Specific Ideas

Add tests that fail if shared benchmark orchestration reaches into actor `actions.hpp`,
`guards.hpp`, or `detail.hpp`, or if direct generation/diarization suite append wiring drifts back
into `bench_runner.cpp`.

</specifics>

<deferred>
## Deferred Ideas

Milestone lifecycle audit and archive happen after this phase commit.

</deferred>
