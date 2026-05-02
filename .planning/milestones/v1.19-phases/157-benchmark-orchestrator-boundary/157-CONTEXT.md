# Phase 157: Benchmark Orchestrator Boundary - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish the first `tools/bench` shared orchestrator boundary by moving process CLI/config,
asset metadata, request setup, and result/report flow out of the process shim and into a
runner-owned entrypoint. This phase must not change benchmark semantics, case registration, CMake
suite layout, dependency manifests, or quality-gate behavior.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- Treat this as pure infrastructure: keep the existing benchmark execution body intact and move it
  behind `emel::bench::run_bench_cli(...)`.
- Preserve current CLI flags, environment variables, output formats, and exit codes.
- Add source-level tests proving `bench_main.cpp` is only a shim and common orchestration lives in
  the runner boundary.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/bench/bench_main.cpp` currently owns CLI mode parsing, environment config, result
  printing, generation/diarization JSONL branching, compare output, and static case lists.
- `tools/bench/bench_runner_tests.cpp` already runs the benchmark binary and contains source tests
  for actor-boundary constraints.

### Established Patterns
- Tool-boundary refactors in v1.18 kept the process `main` as a small shim and moved behavior to
  runner-owned entrypoints.
- Focused doctest source checks are the existing guardrail for tool ownership regressions.

### Integration Points
- `tools/bench/CMakeLists.txt` owns the `bench_runner` target source list and focused benchmark
  tests.

</code_context>

<specifics>
## Specific Ideas

No specific requirements beyond the roadmap and issue #55 constraints.

</specifics>

<deferred>
## Deferred Ideas

Runner contracts, discovery metadata, independent build targets, dependency manifests, and
quality-gate consumption remain deferred to later v1.19 phases.

</deferred>
