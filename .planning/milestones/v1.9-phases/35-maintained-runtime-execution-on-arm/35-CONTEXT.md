# Phase 35: Reference Comparators And Smoke Proof - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 35 proves the matched `llama.cpp` reference executable and the shared runtime-smoke boundary
for the maintained Qwen3 executable-size slice. The phase covers final executable builds for both
rows and the local smoke run on the same `hello` -> first-token path.

The phase does not widen comparator scope beyond the EMEL row and one `llama.cpp` reference row.

</domain>

<decisions>
## Implementation Decisions

### Comparator Truth
- **D-01:** The reference row is one final linked `llama.cpp` executable for the same maintained
  fixture and request slice.
- **D-02:** The published comparison remains limited to EMEL and the matched `llama.cpp`
  executable.

### Smoke Proof
- **D-03:** When the maintained fixture is present locally, every published row must pass the same
  `hello` -> first-token smoke path.
- **D-04:** The smoke proof is part of publication truth, not an optional side-check.

### Scope Guardrails
- **D-05:** No additional comparator runtimes in v1.8.
- **D-06:** No claim widening from smoke success into whole-product feature parity.

</decisions>

<specifics>
## Specific Ideas

- The milestone already has a working local reference probe path, so the main task is to prove and
  record it under the maintained workload boundary.

</specifics>

<canonical_refs>
## Canonical References

- `AGENTS.md`
- `docs/rules/sml.rules.md`
- `.planning/PROJECT.md`
- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`
- `scripts/embedded_size.sh`
- `tools/embedded_size/reference_probe/main.cpp`
- `tools/embedded_size/emel_probe/main.cpp`

</canonical_refs>

<code_context>
## Existing Code Insights

- The current embedded-size tool already builds both final probe executables.
- Local smoke runs have already been observed to succeed for the maintained fixture.
- The gap is auditability and phase proof, not comparator-path existence.

</code_context>

<deferred>
## Deferred Ideas

- Additional comparator runtimes
- Broader model or request matrices

</deferred>

---
*Phase: 35-maintained-runtime-execution-on-arm*
*Context gathered: 2026-04-02*
