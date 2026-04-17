---
phase: 52
slug: shared-embedding-session
created: 2026-04-14
status: ready
---

# Phase 52 Context

## Phase Boundary

Phase 52 closes the gap between the now-live text, vision, and audio embedding lanes by proving
they already land on one deterministic shared-session contract. The scope is narrow: keep the
existing `src/emel/embeddings/generator/` actor, prove uniform output publication and truncation
behavior across modalities, and confirm the first maintained TE slice remains synchronous and
limited to one modality per request. This phase does not widen into golden baselines, cross-modal
ranking claims, batching, or public API work.

## Implementation Decisions

### Shared Contract
- Keep one embedding-session actor under `src/emel/embeddings/generator/`; do not fork new
  modality-specific orchestration machines in this phase.
- Treat Phase 52 as contract consolidation and proof, not a structural runtime rewrite.
- Preserve explicit modality routing in the state machine and prove that all three request surfaces
  terminate through the same publication/truncation path.

### Maintained Request Surface
- Keep the maintained slice synchronous and bounded with one modality per top-level request.
- Prove uniform supported truncation at `1280`, `768`, `512`, `256`, and `128`.
- Prove uniform invalid truncation rejection on one unsupported dimension instead of adding ad hoc
  modality-local validation rules.

### Scope Guardrails
- Reuse the canonical in-memory proof anchors already locked in Phase 47.
- Limit code churn to shared test scaffolding and proof coverage unless the contract itself proves
  inconsistent.
- Defer upstream golden-vector parity, canonical audio-anchor fidelity cleanup, and cross-modal
  semantic assertions to Phase 53.

## Existing Code Insights

### Reusable Assets
- `src/emel/embeddings/generator/detail.hpp` already centralizes shared embedding publication,
  truncation, and normalization.
- `tests/embeddings/text_embedding_lane_tests.cpp`,
  `tests/embeddings/vision_embedding_lane_tests.cpp`, and
  `tests/embeddings/audio_embedding_lane_tests.cpp` already prove each modality lane independently.
- `tests/embeddings/te_fixture.hpp` already owns the maintained TE loader path and the canonical
  in-memory payload synthesis helpers used by the modality tests.

### Contract Shape To Prove
- Text, image, and audio requests already land on one output surface:
  `embedding buffer + output dimension + error_out + optional callbacks`.
- Supported Matryoshka truncation dimensions are already shared across modalities.
- The actor remains one-request-at-a-time and one-modality-per-request because each public event is
  still a separate top-level dispatch surface.

## Specific Ideas

- Add a dedicated shared-contract test file that runs the canonical text/image/audio anchors
  through the same embedding session and checks:
  - shared normalization
  - shared supported truncation dimensions
  - shared unsupported-dimension rejection
- Extract reusable fixture helpers for the canonical in-memory image/audio anchors so Phase 52
  proof and later golden tests share one synthesis path.
- Keep verification focused on the embedding test family plus full repo quality gates.

## Deferred Ideas

- stored upstream golden embeddings
- semantic cross-modal similarity/ranking thresholds
- public embedding API or batching
- `text/forward`, `vision/forward`, or `audio/forward` domain extraction

## Primary Sources

- `src/emel/embeddings/generator/detail.hpp`
- `src/emel/embeddings/generator/sm.hpp`
- `tests/embeddings/te_fixture.hpp`
- `tests/embeddings/shared_embedding_session_tests.cpp`
- `tests/embeddings/fixtures/te75m/README.md`

---
*Phase: 52-shared-embedding-session*
*Context gathered: 2026-04-14*
