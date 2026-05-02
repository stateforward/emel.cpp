# Phase 173: SML Migration Evidence Reconstruction - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 173 reconstructs the missing source-backed evidence for v1.20 dependency pin and source
namespace migration requirements. It does not change runtime behavior; it records the live
provenance and scan evidence needed to close DEP-01, DEP-02, DEP-03, SRC-01, and SRC-02.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- Treat this as an infrastructure/evidence phase.
- Use live git history and source scans as the primary evidence.
- Do not broaden the milestone scope beyond the existing SML dependency and namespace migration.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cmake/sml_version.cmake` owns the SML dependency pin.
- Root `CMakeLists.txt` and `tools/docsgen/CMakeLists.txt` consume the shared pin through
  FetchContent.
- `.planning/v1.20-MILESTONE-AUDIT.md` lists the missing evidence and source-backed gaps.

### Established Patterns
- Prior closeout phases record evidence in SUMMARY, VERIFICATION, and VALIDATION artifacts.
- Requirements are checked off only after live source-backed evidence exists.

### Integration Points
- `.planning/REQUIREMENTS.md` traceability maps DEP/SRC requirements to this closure phase.
- `.planning/ROADMAP.md` records this phase as the gap closure phase for dependency and source
  migration evidence.

</code_context>

<specifics>
## Specific Ideas

No specific requirements beyond the audit findings.

</specifics>

<deferred>
## Deferred Ideas

Runtime behavior proof, documentation path repair, guardrail wiring, and final closeout are deferred
to phases 174-177.

</deferred>

