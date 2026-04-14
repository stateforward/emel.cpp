---
phase: 47
slug: te-truth-anchor
created: 2026-04-13
status: ready
---

# Phase 47 Context

## Phase Boundary

Phase 47 locks the maintained TE truth surface before runtime work broadens. This phase is about
fixture identity, provenance, and proof-input discipline, not about adding `omniembed` execution
or modality runtime code yet.

The maintained surface for this milestone is one exact local GGUF file under `tests/models/`, one
exact upstream truth source, and one tiny deterministic proof corpus expressed as in-memory
payload contracts. It must not imply support for sibling quantizations, generic media decoding, or
broader `omniembed` coverage.

## Implementation Decisions

### Fixture Truth
- The maintained TE slice is fixed to `tests/models/TE-75M-q8_0.gguf`.
- Fixture metadata should follow current upstream Hugging Face GGUF/model truth, not stale local
  assumptions or broader family claims.
- The repo should explicitly name the unapproved sibling `TE-75M-q5_0.gguf` only as an upstream
  sibling, not as maintained support.

### Proof Corpus Truth
- The canonical proof corpus should stay tiny and deterministic.
- Proof inputs should be defined as in-memory payload contracts, not as generic PNG/JPEG/WAV
  decoding requirements.
- The smallest useful corpus is a pairwise anchor set: one text-image anchor and one text-audio
  anchor.

### Scope Discipline
- Keep Phase 47 focused on fixture and corpus truth surfaces: `tests/models`, a proof-corpus
  manifest, and regression checks that keep those anchors from drifting.
- Do not start `src/emel/model/omniembed`, `text/encoders`, `vision/encoders`, `audio/encoders`,
  or `embeddings/generator` implementation work in this phase.
- State-machine structure changes remain deferred to later phases and require the existing AGENTS
  approval gate before execution.

## Existing Code Insights

### Reusable Assets
- `tests/models/README.md` already acts as the repo-visible maintained fixture registry.
- `tests/model/*` already holds loader-oriented doctest coverage, which is a clean home for a
  truth-anchor regression test.
- `.planning/PROJECT.md`, `.planning/REQUIREMENTS.md`, and `.planning/ROADMAP.md` already lock the
  milestone to one narrow maintained TE slice.

### Established Patterns
- Prior fixture-lock phases document one exact model path, checksum, source, and download URL
  before runtime expansion.
- Repo quality-gate proof prefers small deterministic file/assertion tests over ad hoc notes.
- Maintained claims must stay additive and explicit rather than implying family-wide support.

### Integration Points
- `tests/models/README.md` is the maintained fixture truth surface.
- `tests/embeddings/fixtures/` is the narrow place to define TE proof inputs without widening
  runtime architecture yet.
- `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`, and `.planning/STATE.md` need phase-closeout
  updates once the truth anchor lands.

## Specific Ideas

- Use the current Hugging Face TE GGUF metadata as the maintained truth source for Phase 47.
- Lock a pairwise proof corpus instead of inventing a forced three-way semantic triplet.
- Keep the corpus simple enough that later phases can synthesize payloads in memory with no file
  decode dependency.

## Deferred Ideas

- `q5_0` runtime support
- any `omniembed` model-contract code
- image/audio preprocessing implementation
- shared embedding session or public API work

---
*Phase: 47-te-truth-anchor*
*Context gathered: 2026-04-13*
