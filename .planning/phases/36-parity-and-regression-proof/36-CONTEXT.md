# Phase 36: Snapshot And README Publication - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 36 covers the executable-size publication plumbing for v1.8: the script output schema, the
checked-in snapshot, the generated README section, and the wording that explains the scope and
non-claims of the measurement.

This phase is about publication infrastructure. It is not the final closeout refresh; later gap
closure may still need to refresh stale evidence before milestone audit can pass.

</domain>

<decisions>
## Implementation Decisions

### Publication Truth
- **D-01:** `scripts/embedded_size.sh` is the maintained source of executable-size metadata.
- **D-02:** `snapshots/embedded_size/summary.txt` is the stored publication artifact.
- **D-03:** `README.md` must be generated from that snapshot through docsgen/template plumbing.

### Wording Guardrails
- **D-04:** Publication text must say this is a matched Qwen3 E2E executable comparison.
- **D-05:** Publication text must not imply whole-product parity.

### Scope Guardrails
- **D-06:** The publication surface remains narrow to the canonical Qwen3 fixture and bounded
  request slice.

</decisions>

<specifics>
## Specific Ideas

- The milestone already has the script and docs pipeline wired; the risk is drift between the local
  truth and the checked-in publication evidence.

</specifics>

<canonical_refs>
## Canonical References

- `AGENTS.md`
- `docs/rules/sml.rules.md`
- `.planning/PROJECT.md`
- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`
- `scripts/embedded_size.sh`
- `tools/docsgen/docsgen.cpp`
- `docs/templates/README.md.j2`
- `snapshots/embedded_size/summary.txt`

</canonical_refs>

<code_context>
## Existing Code Insights

- The docs pipeline is already in place.
- The current checked-in snapshot is stale relative to the latest local executable-size run.
- This phase should therefore prove the plumbing itself, while later closeout work refreshes the
  final publication evidence.

</code_context>

<deferred>
## Deferred Ideas

- Bundle-size publication
- Blocking size gates
- Broader publication matrices

</deferred>

---
*Phase: 36-parity-and-regression-proof*
*Context gathered: 2026-04-02*
