# Phase 203 Context: Closeout State And Rule Debt Cleanup

## Source

Phase 203 was created from `.planning/v1.23-MILESTONE-AUDIT.md` after the follow-up v1.23 audit
returned `tech_debt`.

## Goal

Close the remaining non-blocking audit debt so v1.23 can be archived without accepting
contradictory planning state, superseded proof wording, stale scaffold markers, or known
source-rule debt.

## Required Repairs

1. Align active and archived planning artifacts so Phase 201 is clearly historical/superseded and
   Phase 202 remains the active VAL-01/02/03 proof.
2. Reconcile `.planning/PROJECT.md`, `.planning/ROADMAP.md`, and `.planning/STATE.md` so they agree
   whether v1.23 is pending final closeout or archived.
3. Repair or explicitly reconcile the older `model/tensor` persistent context naming debt without
   moving tensor residency ownership out of `model/tensor`.
4. Remove or rewrite IO scaffold benchmark comments so they cannot be mistaken for concrete
   strategy benchmark claims.
5. Regenerate and commit any maintained docs, lint snapshots, benchmark snapshots, benchmark
   outputs, model artifacts, or fixtures required by changed files.

## Validation Expectations

- `scripts/check_domain_boundaries.sh`
- focused IO/model CTest lanes
- docs generation / architecture regeneration if documentation inputs change
- lint snapshots if formatting snapshots change
- benchmark lanes and snapshots when changed files make them relevant
- changed-file scoped `scripts/quality_gates.sh`
- final `$gsd-audit-milestone`

## User Direction

The user explicitly approved updating model artifacts, snapshots, and benchmarks if needed to do
the repair correctly.
