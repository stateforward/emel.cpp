# Phase 78: Generation Workload Discovery Cutover - Context

**Gathered:** 2026-04-21
**Status:** Completed

<domain>
## Phase Boundary

Replace the generation benchmark's hard-coded workload manifest array with deterministic discovery
from checked-in workload manifests.

</domain>

<decisions>
## Implementation Decisions

### Discovery
- Discover `tools/bench/generation_workloads/*.json` in sorted path order.
- Load and validate each manifest through the existing generation workload parser.
- Keep workload filters compatible with ID, case name, and compare group.

### the agent's Discretion
Keep the existing `generation_compare/v1` record shape unchanged.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `load_generation_workload_manifest` already validates one manifest plus its prompt fixture.

### Established Patterns
- `generation_bench.cpp` builds `generation_case_spec` values before running EMEL/reference lanes.

### Integration Points
- `generation_bench.cpp`
- `generation_workload_manifest.hpp`
- `bench_runner_tests.cpp`

</code_context>

<specifics>
## Specific Ideas

Adding a generation workload should require a prompt fixture only when needed and one workload
manifest.

</specifics>

<deferred>
## Deferred Ideas

New generator model/runtime support is not part of this phase.

</deferred>
