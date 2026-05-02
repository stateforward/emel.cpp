# Phase 183: Parallel Orchestration And Reporting - Context

**Gathered:** 2026-05-02
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase runs independent quality-gate lanes concurrently only after the serial gate preflight and
zig build complete. It covers lane process orchestration, ordered log replay, timing capture, and
exit status aggregation.

</domain>

<decisions>
## Implementation Decisions

### Parallel Execution
- Keep domain boundary, legacy SML surface, and zig build serial.
- Run benchmark, coverage, parity, and fuzz lanes in a parallel group because they use separate
  build or runtime artifacts after the main build has completed.
- Control parallel behavior with `EMEL_QUALITY_GATES_PARALLEL`, defaulting to `auto`.
- Preserve a serial fallback with `EMEL_QUALITY_GATES_PARALLEL=never`.

### Reporting
- Capture each lane into a temporary log file and replay logs in deterministic lane order.
- Record duration for every lane, including failed lanes.
- Preserve benchmark failure routing through `bench_status`.
- Return non-benchmark failures immediately after the parallel group.

### the agent's Discretion
Child lanes should suppress timing snapshot writes and let the parent write the aggregate timing
snapshot to avoid concurrent writes to the same file.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `run_step` and `run_step_allow_fail` already measure lane duration and update timing snapshots.
- Benchmark failures already flow through `bench_status`.
- Coverage, parity, benchmark, and fuzz lanes are already factored into separate shell functions.

### Established Patterns
- Environment variables control quality-gate policy.
- Shell lane helpers are explicit and source-readable.
- Timing output is kept in `snapshots/quality_gates/timing.txt`.

### Integration Points
- `scripts/quality_gates.sh`
- `tools/bench/quality_gates_tests.cpp`
- `snapshots/quality_gates/timing.txt`

</code_context>

<specifics>
## Specific Ideas

Parallelism must improve wall time without making logs ambiguous or corrupting shared timing
artifacts.

</specifics>

<deferred>
## Deferred Ideas

CI-native distributed execution remains out of scope.

</deferred>
