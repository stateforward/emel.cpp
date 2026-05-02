# Phase 181: Manifest Impact Resolution - Context

**Gathered:** 2026-05-02
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase resolves impacted parity and benchmark runners from checked-in dependency manifests,
with conservative full-runner fallback for missing, stale, malformed, uncertain, or unmatched
states.

</domain>

<decisions>
## Implementation Decisions

### Manifest Inputs
- Use `tools/paritychecker/dependency_manifest.txt` as the parity manifest baseline.
- Continue using `tools/bench/dependency_manifest.txt` as the benchmark manifest baseline.
- Resolve changed files from `EMEL_QUALITY_GATES_CHANGED_FILES` or the existing git-based
  changed-file collection path.
- Preserve full fallback when manifest freshness cannot be proved by the maintained tool binary.

### Conservative Fallbacks
- Missing parity manifest selects full parity.
- Unmatched parity-relevant paths select full parity.
- Missing or stale benchmark manifests select full benchmark.
- Benchmark tool changes not covered by the manifest select full benchmark.

### the agent's Discretion
Manifest resolution may reuse the existing benchmark helper shape so parity and benchmark behavior
stay auditable in one script.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `bench_dependency_manifest_apply_changed_files()` already maps benchmark manifest records to
  runner selections.
- `bench_dependency_manifest_requires_full_gate()` already asks `bench_runner` to emit and check a
  manifest.
- `tools/paritychecker/dependency_manifest.txt` provides parity runner-to-path records.

### Established Patterns
- Manifest records use `record runner=<name> ... path=<path> ...`.
- Uncertain freshness states return to full gate execution.
- Runner decisions are echoed to stderr so logs explain why work was selected.

### Integration Points
- `scripts/quality_gates.sh`
- `tools/paritychecker/dependency_manifest.txt`
- `tools/bench/dependency_manifest.txt`
- `tools/bench/quality_gates_tests.cpp`

</code_context>

<specifics>
## Specific Ideas

Issue #58 specifically requires selective quality gates after prior benchmark and parity manifest
work lands, so impact resolution must trust checked-in manifests only when freshness checks pass.

</specifics>

<deferred>
## Deferred Ideas

Extending this manifest approach to future tool families remains deferred.

</deferred>
