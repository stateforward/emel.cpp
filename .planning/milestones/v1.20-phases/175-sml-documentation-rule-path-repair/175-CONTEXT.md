# Phase 175: SML Documentation Rule Path Repair - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 175 repairs active contributor guidance that still referenced the stale `docs/sml.rules.md`
path and proves docs/examples/planning guidance consistently use `docs/rules/sml.rules.md` and the
migrated `stateforward::sml` surface.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- Keep the repair narrow to active guidance path conflicts.
- Do not rewrite archival third-party SML reference content unless it conflicts with active
  contributor guidance.
- Verify with direct grep, docs generation, and lint snapshot checks.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `AGENTS.md` is active contributor guidance.
- `docs/plans/rearchitecture.plan.md` is active planning guidance.
- `docs/rules/sml.rules.md` is the normative SML rules document.

### Established Patterns
- Documentation checks run through `scripts/generate_docs.sh --check`.
- Lint snapshot checks run through `scripts/lint_snapshot.sh`.

### Integration Points
- `.planning/v1.20-MILESTONE-AUDIT.md` identified stale `docs/sml.rules.md` references as a
  closeout blocker.

</code_context>

<specifics>
## Specific Ideas

No user-facing behavior. This is a documentation consistency repair.

</specifics>

<deferred>
## Deferred Ideas

Guardrail enforcement is deferred to Phase 176.

</deferred>

