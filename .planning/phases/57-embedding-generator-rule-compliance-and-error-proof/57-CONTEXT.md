---
phase: 57
slug: embedding-generator-rule-compliance-and-error-proof
created: 2026-04-15
status: ready
---

# Phase 57 Context

## Phase Boundary

Phase 57 closes the two blocking audit findings left after the `v1.11` reopen work: runtime
branching still lives inside maintained embedding-generator actions, and the contract-drift
initialize regression still proves only "some failure" instead of the exact maintained error class.
This phase stays narrowly scoped to `src/emel/embeddings/generator` plus the regression test that
audits that surface.

## Implementation Decisions

### State-Machine Scope
- Keep the existing embedding-generator transition graph intact unless the current graph makes the
  fix impossible.
- Move success and failure bookkeeping into runtime-event fields and existing guards where possible.
- Do not broaden the TE runtime contract or public embedding API.

### Rule-Compliance Scope
- Remove explicit runtime `if` branching from the maintained image/audio prepare actions and the
  maintained text/image/audio execution actions.
- Remove explicit runtime `if` branching from maintained embedding publication actions without
  reintroducing hidden control flow in helper functions called from those actions.
- Keep RTC behavior and dispatch-time allocation rules unchanged.

### Error-Proof Scope
- Make initialize preserve constructor-time contract drift as `model_invalid` instead of collapsing
  into generic invalid-request handling.
- Prove the missing-family regression with an exact error assertion.
- Preserve the current maintained initialize behavior for valid TE fixtures and ordinary bad user
  requests.

## Existing Code Insights

### Relevant Surfaces
- `src/emel/embeddings/generator/actions.hpp` still contains audit-flagged runtime `if` branching.
- `src/emel/embeddings/generator/guards.hpp` already has decision guards for initialize, prepare,
  execute, and publish phases.
- `src/emel/embeddings/generator/detail.hpp` reserves all runtime buffers before dispatch and
  already resets runtime readiness when the maintained contract is broken.
- `tests/embeddings/shared_embedding_session_tests.cpp` already reproduces the missing-family
  initialize case, but only checks for non-`none` failure.

### Constraints
- AGENTS requires no dispatch-time allocation and no hidden runtime path selection inside actions
  or helpers they call.
- AGENTS also requires asking before changing state-machine structure, so prefer a guard/event-data
  solution over graph expansion.
- The worktree is already dirty, so edits must stay surgical and avoid reverting unrelated user
  changes.

## Specific Ideas

- Relax initialize admission so constructor-time runtime reservation failures can flow through the
  existing initialize decision states and emerge as `model_invalid` or backend instead of
  `invalid_request`.
- Add explicit runtime-event booleans for embedding execution success so completion guards, not
  actions, choose the next path.
- Make publish actions operate on already-validated dimensions and output buffers so the action body
  performs bounded data movement only.

## Deferred Ideas

- Broader embedding-generator graph refactors
- Additional benchmark publication work for the maintained TE slice
- Any public API or multimodal capability expansion beyond the shipped TE contract

---
*Phase: 57-embedding-generator-rule-compliance-and-error-proof*
*Context gathered: 2026-04-15*
