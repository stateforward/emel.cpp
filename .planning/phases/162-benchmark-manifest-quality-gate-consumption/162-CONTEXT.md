# Phase 162: Benchmark Manifest Quality-Gate Consumption - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the benchmark dependency manifest into changed-file scoped quality-gate selection. This phase
should consume the Phase 161 manifest conservatively without changing benchmark runner behavior or
manifest schema.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- Reuse the parity manifest freshness pattern for benchmark manifest checks.
- Treat missing, stale, uncertain, or unparseable benchmark manifest state as a full benchmark
  trigger.
- Use `runner=all` records for shared benchmark inputs and per-runner records for scoped suite
  selection.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/quality_gates.sh` already has parity manifest write/check freshness handling.
- `tools/bench/dependency_manifest.txt` now records shared and per-runner benchmark inputs.
- `tools/bench/quality_gates_tests.cpp` provides source-level regression tests for quality-gate
  behavior.

### Integration Points
- `infer_quality_gate_scope` should call manifest-based benchmark inference after collecting
  changed files.
- `run_benchmark_gates` should check manifest freshness before deciding whether to skip, run
  scoped suites, or run the full benchmark gate.

</code_context>

<specifics>
## Specific Ideas

Keep the local validation scoped to `generation` because this phase changes gate logic and source
tests, not benchmark runtime behavior.

</specifics>

<deferred>
## Deferred Ideas

Behavior and lane-isolation closure remains Phase 163.

</deferred>
