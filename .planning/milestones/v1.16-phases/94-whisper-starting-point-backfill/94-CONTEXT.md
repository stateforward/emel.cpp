# Phase 94: Whisper Starting Point Backfill - Context

**Gathered:** 2026-04-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Audit the already-started Whisper local changes and correct milestone framing so the repo starts
`v1.16` from truthful, source-backed state without over-claiming parity or runtime completion.

</domain>

<decisions>
## Implementation Decisions

### Backfill Audit Scope
- Classify current Whisper-adjacent local changes into exactly four buckets:
  landed, keep-and-fix, replace, or discard.
- Treat this phase as source-of-truth auditing and contract correction only; do not broaden into
  runtime completion, parity closure, or benchmark claims.

### Contract And Wording Corrections
- Replace q80-only wording with maintained variant-family wording where phase-appropriate while
  preserving fixture-level facts for `model-tiny-q80.gguf`.
- Make loader/model-contract support wording explicit as loader-only and non-parity, non-runtime.

### Kernel Compliance Review
- Review started kernel quantized edits for `AGENTS.md` and `docs/rules/sml.rules.md` control-flow
  compliance before any Phase 96 expansion.
- Record review findings as source-backed evidence instead of inferred status from planning docs.

### the agent's Discretion
Use concise, source-backed audit wording and file-level classifications so downstream planning can
execute Phase 95+ with clear keep/fix/replace boundaries.

</decisions>

<canonical_refs>
## Canonical References

### Milestone Scope And Requirements
- `.planning/ROADMAP.md` — Phase 94 goal and success criteria.
- `.planning/REQUIREMENTS.md` — `BACK-01`, `BACK-02`, and `BACK-03` requirements.
- `.planning/PROJECT.md` — v1.16 milestone constraints and no-overclaim policy.

### Engineering Rules
- `AGENTS.md` — authoritative engineering contract for behavior routing and performance policy.
- `docs/rules/sml.rules.md` — Stateforward.SML invariants and explicit routing requirements.

### Current Whisper Starting Material
- `src/emel/model/whisper/detail.hpp` — Whisper model contract types and interfaces.
- `src/emel/model/whisper/detail.cpp` — Whisper architecture validation and hyperparameter loading.
- `src/emel/model/architecture/detail.cpp` — architecture registry wiring.
- `src/emel/model/data.hpp` and `src/emel/model/data.cpp` — execution architecture helpers and
  contract validation entrypoints.
- `src/emel/kernel/detail.hpp` — native quantized dtype/block/dequant paths under review.
- `tests/model/loader/lifecycle_tests.cpp` — Whisper loader/contract lifecycle tests.
- `tests/model/fixture_manifest_tests.cpp` and `tests/models/README.md` — fixture provenance and
  wording truth.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `emel::model::resolve_architecture()` and architecture table registration pattern in
  `src/emel/model/architecture/detail.cpp`.
- Existing model-contract validation pattern used by other architectures in
  `src/emel/model/*/detail.cpp`.
- Fixture manifest enforcement pattern in `tests/model/fixture_manifest_tests.cpp`.

### Established Patterns
- Architecture loading and validation are isolated in model-family `detail.cpp` files.
- Fixture truth is documented under `tests/models/README.md` and asserted by doctest coverage.
- Quantized kernel primitives live in `src/emel/kernel/detail.hpp` and should remain data-plane.

### Integration Points
- Whisper architecture registration and execution-contract helpers in `src/emel/model/**`.
- Fixture wording and provenance checks in `tests/models/README.md` and fixture manifest tests.
- Kernel quantized support review handoff into future Phase 96 implementation planning.

</code_context>

<specifics>
## Specific Ideas

Produce an explicit file-by-file classification ledger in phase execution artifacts so Phase 95
planning can consume it directly without re-auditing the same local changes.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within the Phase 94 audit boundary.

</deferred>
