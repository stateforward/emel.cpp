# Phase 176: Legacy SML Guardrail And Quality Gate Repair - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 176 closes the audit blocker for missing legacy SML drift enforcement and weakened
quality-gate wiring. It adds a maintained source check, wires it into `scripts/quality_gates.sh`,
and restores lint snapshot execution without updating snapshots.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- Implement the guardrail as a small Unix shell script using `rg`, consistent with existing repo
  checks.
- Scan active code/docs/tooling paths and exclude archival third-party SML reference material.
- Restore the lint snapshot lane in the main quality gate.
- Let changed-file benchmark inference choose relevant benchmark suites instead of forcing an
  invalid suite name.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/quality_gates.sh` already runs domain boundary, build, coverage, parity, fuzz,
  benchmark, and docs lanes.
- Existing check scripts use `set -euo pipefail` and hard-fail on missing `rg`.

### Established Patterns
- Quality gate lanes are wrapped with `run_step`.
- Irrelevant lanes may skip by changed-file inference, but relevant lanes must run.

### Integration Points
- New guardrail script belongs under `scripts/`.
- `scripts/quality_gates.sh` is the maintained entrypoint for changed-file scoped gates.

</code_context>

<specifics>
## Specific Ideas

No specific requirements beyond the audit findings.

</specifics>

<deferred>
## Deferred Ideas

Final milestone audit and closeout are deferred to Phase 177.

</deferred>

