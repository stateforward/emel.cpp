# Phase 33: Workload Boundary And Claim Guardrails - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 33 locks the maintained executable-size workload and claim boundary for v1.8. The maintained
surface is one canonical fixture, one canonical prompt contract, one bounded generation slice, and
one narrow comparator set: EMEL plus a matched `llama.cpp` reference executable.

This phase is about truth boundaries, not implementation breadth. It defines what the later
runner, smoke, and publication phases are allowed to claim.

</domain>

<decisions>
## Implementation Decisions

### Workload Truth
- **D-01:** The maintained executable-size workload is fixed to
  `tests/models/Qwen3-0.6B-Q8_0.gguf`.
- **D-02:** The maintained request is the structured `hello` prompt on the generated runner path.
- **D-03:** The maintained generation slice is `max_tokens=1`.

### Measurement Boundary
- **D-04:** Final linked executables are the primary truth surface for v1.8.
- **D-05:** Static or shared library artifact size is not a maintained publication claim for this
  milestone.
- **D-06:** The measurement is a matched executable comparison, not a whole-product parity claim.

### Comparator Scope
- **D-07:** The published comparator scope for v1.8 is EMEL versus one matched `llama.cpp`
  reference executable only.
- **D-08:** Additional comparator runtimes are out of scope for v1.8 because they widen the claim
  surface and add planning churn.

### Publication Guardrails
- **D-09:** The README embedded-size section must stay generated from
  `snapshots/embedded_size/summary.txt`.
- **D-10:** Runtime smoke evidence must stay explicit in the published metadata.
- **D-11:** The executable-size surface remains non-blocking in `quality_gates` for now.

</decisions>

<specifics>
## Specific Ideas

- The user explicitly redirected the work away from library sizes and toward a real executable
  measurement for embedded relevance.
- The user also removed LiteRT from the milestone to keep the scope narrow and auditable.
- The milestone should close with one truthful Qwen3 E2E executable-size claim, not a family of
  loosely matched comparisons.

</specifics>

<canonical_refs>
## Canonical References

- `AGENTS.md`
- `docs/rules/sml.rules.md`
- `.planning/PROJECT.md`
- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- `scripts/embedded_size.sh`
- `tools/embedded_size/emel_probe/main.cpp`
- `tools/embedded_size/reference_probe/main.cpp`

</canonical_refs>

<code_context>
## Existing Code Insights

- The executable-size flow already exists locally and builds final EMEL and reference probes.
- The current planning risk is not lack of code. It is lack of traceable proof and stale
  publication evidence.
- Phase 33 should therefore stay small and declarative: lock the boundary and keep later phases
  from widening it.

</code_context>

<deferred>
## Deferred Ideas

- Deployable bundle-size accounting
- Additional comparator runtimes
- Blocking executable-size quality gates
- Broader Qwen or Liquid model-family publication

</deferred>

---
*Phase: 33-fixture-metadata-and-contract-lock*
*Context gathered: 2026-04-02*
