# Phase 160: Benchmark Independent Build Targets - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Make benchmark suite sources build through independent CMake targets or isolated source groups so
runner additions have localized build impact. This phase should not change benchmark selection,
runner registration semantics, output schemas, or dependency-manifest behavior.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- Use per-suite CMake `OBJECT` libraries so each maintained benchmark family has a visible target.
- Keep the `bench_runner` executable as the operator-facing binary and link selected suite object
  files into it.
- Preserve the existing compile-definition based disabled-stub behavior for filtered builds.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 159 localized suite metadata in `bench_runner_registry.cpp`.
- `bench_disabled_cases.cpp` uses `EMEL_BENCH_ENABLE_*` definitions to provide stubs for suites
  that are not compiled into a filtered runner build.

### Established Patterns
- Existing `EMEL_BENCH_SUITE_FILTER` behavior selects one suite at CMake configure time.
- Focused source checks in `bench_runner_tests.cpp` already prove benchmark tool ownership
  boundaries.

### Integration Points
- `tools/bench/CMakeLists.txt` should create suite-owned build targets inside
  `add_bench_runner_suite(...)`.
- `bench_runner_tests.cpp` should prove suite sources flow through independent object targets.

</code_context>

<specifics>
## Specific Ideas

No placeholder runner is needed if the maintained suite list itself proves independent object
targets and source ownership.

</specifics>

<deferred>
## Deferred Ideas

Dependency manifests and quality-gate manifest consumption remain deferred to Phases 161 and 162.

</deferred>
