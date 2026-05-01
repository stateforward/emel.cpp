# Phase 164: Benchmark Serialized Runner Process Seam Closure - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning
**Mode:** Autonomous smart discuss

<domain>
## Phase Boundary

Close `RUNNER-02` by making the existing `bench_runner_request/v1` and
`bench_runner_result/v1` serialized contract part of a live benchmark runner process entrypoint.
The phase must preserve existing `bench_runner` CLI behavior and manifest operations while adding a
fail-closed process seam that can be exercised through the maintained benchmark binary.

</domain>

<decisions>
## Implementation Decisions

### Process Seam Shape
- Add a production `bench_runner` mode that reads a serialized request from a file and writes a
  serialized result to a file.
- Keep benchmark stdout/stderr behavior separate from the serialized result file so existing output
  schemas remain stable.
- Prevalidate malformed payloads, unknown modes, conflicting JSONL flags, and unknown suites before
  dispatching runner execution.
- Preserve current in-process CLI behavior for ordinary operator invocations.

### Verification Strategy
- Add focused `bench_runner_tests` coverage that invokes the built `bench_runner` binary with the
  serialized process flags.
- Use a one-run maintained suite request for success-path process-seam proof; the local filtered
  benchmark build is generation-scoped, so the success test targets the compiled generation runner
  while the full unfiltered runner test suite remains the broad behavior proof.
- Add fail-closed tests for malformed payloads and unknown suites.
- Keep dependency-manifest freshness and quality-gate behavior unchanged.

### the agent's Discretion
- Choose exact flag names and internal helper boundaries to fit the existing `bench_runner.cpp`
  style, as long as the live process seam is source-backed and test-covered.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/bench/bench_runner_contract.hpp` already defines `runner_request`, `runner_result`,
  `serialize_runner_request`, `parse_runner_request`, `serialize_runner_result`, and
  `parse_runner_result`.
- `tools/bench/bench_runner.cpp` already normalizes CLI config into `runner_request` before
  dispatching registered runner spans.
- `tools/bench/bench_runner_tests.cpp` already has helpers for invoking the built `bench_runner`
  binary and reading temporary output files.

### Established Patterns
- Manifest operations are parsed as dedicated exclusive CLI modes before normal benchmark
  execution.
- Tests prefer source-backed checks plus binary invocation through `BENCH_RUNNER_BINARY_PATH`.
- Unknown benchmark suites fail closed instead of silently skipping.

### Integration Points
- `run_bench_cli(...)` is the production process entrypoint used by `bench_main.cpp`.
- `default_runner_cases()` and `kernel_runner_cases()` are the registered runner surfaces.
- `bench_runner_tests` is the maintained focused verification target for benchmark runner behavior.

</code_context>

<specifics>
## Specific Ideas

Use the existing serialized contract rather than defining a new schema.

</specifics>

<deferred>
## Deferred Ideas

Out-of-repo or foreign-language runner packaging remains future scope; this phase only needs a
repo-owned live process seam.

</deferred>
