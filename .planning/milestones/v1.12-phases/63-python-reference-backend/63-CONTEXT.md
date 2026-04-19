---
phase: 63
slug: python-reference-backend
created: 2026-04-17
status: complete
---

# Phase 63 Context

## Phase Boundary

Phase 63 adds the first Python reference backends on top of the shared compare contract. It stays
tool-local and deterministic: the maintained backend uses stored upstream TE goldens, while an
optional live TE backend reports explicit errors if the Python ML stack is unavailable.

## Implementation Decisions

### Scope
- Add a maintained Python backend that emits stored upstream TE goldens under the shared schema.
- Add an optional live Python TE backend path with explicit error records for missing script,
  dependency, or execution failures.
- Keep the compare surface identical to the C++ and EMEL lanes.

### Constraints
- Do not let Python backend failures corrupt or mask the EMEL lane.
- Keep the maintained backend runnable without requiring the full upstream Python model stack.

## Existing Code Insights

- The repo already had stored TE goldens and a reproducible Python generator in
  `tests/embeddings/fixtures/te75m`.
- The missing piece was a backend runner that emitted the shared compare contract instead of
  ad hoc text files.
