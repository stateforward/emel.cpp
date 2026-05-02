# Phase 158: Benchmark Runner Contract And Process Seam - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Define the narrow orchestrator-to-runner request/result contract for `tools/bench` and add a
serialized process-seam payload that future out-of-process or foreign-language runners can use.
This phase must not move case discovery, split CMake targets, add manifests, or change maintained
benchmark behavior.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- Add the contract as a `tools/bench` header so both the runner and focused tests can consume it.
- Keep the serialized process seam deliberately small, deterministic, and local: newline-delimited
  `key=value` fields with explicit schema strings for requests and results.
- Wire the existing runner to construct a normalized request object from current CLI/env inputs
  while preserving all execution branches.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 157 introduced `emel::bench::run_bench_cli(...)` as the runner-owned orchestration entry.
- `bench_common.hpp` already owns the shared benchmark `config` type used by every suite.

### Established Patterns
- Tool contracts in this repo favor small text schemas and explicit schema identifiers over
  hidden C++ object sharing.
- Tests already use source checks and process capture around `bench_runner`.

### Integration Points
- `tools/bench/bench_runner.cpp` should create the request shape before branching into existing
  EMEL/reference/compare execution.
- `tools/bench/bench_runner_tests.cpp` should cover serialization, round-trip parsing, and
  malformed payload rejection.

</code_context>

<specifics>
## Specific Ideas

No specific requirements beyond issue #55's process-level runner seam requirement.

</specifics>

<deferred>
## Deferred Ideas

Runner discovery/registration, build-target separation, dependency manifests, and quality-gate
manifest consumption remain deferred to later phases.

</deferred>
