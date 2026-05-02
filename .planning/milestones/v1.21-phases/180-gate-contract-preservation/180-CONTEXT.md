# Phase 180: Gate Contract Preservation - Context

**Gathered:** 2026-05-02
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase preserves `scripts/quality_gates.sh` as the mandatory top-level validation contract
before selective runner optimization changes are trusted. It covers lane preservation, visible
failure handling, and conservative behavior when the gate script itself changes.

</domain>

<decisions>
## Implementation Decisions

### Gate Contract
- Keep `scripts/quality_gates.sh` as the single maintained quality-gate entrypoint.
- Preserve the existing mandatory serial preflight lanes: domain boundaries, legacy SML surface,
  and zig build.
- Treat changes to `scripts/quality_gates.sh` as gate-contract changes that require conservative
  coverage, parity, benchmark, fuzz, docs, and lint behavior.
- Keep benchmark failures visible and non-silent unless the existing explicit benchmark-regression
  override is set.

### Lane Enforcement
- Keep coverage, lint snapshot, docs, fuzz smoke, parity, benchmark, and lane-isolation gates in
  the script.
- Allow changed-file scope to skip irrelevant lanes only when the script's existing policy and
  manifests can explain why.
- Fail hard when required tools or required lane commands fail.
- Add focused static tests to prevent future weakening of the gate contract.

### the agent's Discretion
The implementation may strengthen conservative fallback for gate-script changes because that file
controls all lane selection behavior.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/quality_gates.sh` already provides `run_step`, `run_step_allow_fail`, timing snapshots,
  changed-file inference, coverage, benchmark, parity, fuzz, lint, and docs lanes.
- `tools/bench/quality_gates_tests.cpp` already uses source-backed static tests for quality-gate
  script contracts.
- `scripts/bench.sh --test-tools` provides a maintained bench-tool validation path.

### Established Patterns
- Gate policy lives in shell functions with explicit environment-variable overrides.
- Benchmark gate failures are captured through `run_step_allow_fail` so the explicit regression
  override can decide whether to stop.
- Tests verify maintained script contracts by reading checked-in source.

### Integration Points
- `scripts/quality_gates.sh`
- `tools/bench/quality_gates_tests.cpp`
- `scripts/bench.sh --test-tools`

</code_context>

<specifics>
## Specific Ideas

Use GitHub issue #58 as the source: optimize the gate without weakening the maintained validation
contract created by the prior milestone.

</specifics>

<deferred>
## Deferred Ideas

Historical dashboards and CI-native distributed execution remain out of scope for this milestone.

</deferred>
