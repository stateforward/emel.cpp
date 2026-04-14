---
phase: 48
slug: omniembed-model-contract
created: 2026-04-14
status: ready
---

# Phase 48 Context

## Phase Boundary

Phase 48 adds truthful `omniembed` model acceptance and a narrow internal execution-contract
surface for the maintained multimodal embedding architecture. This phase is about recognizing the
architecture, loading its core dimensions, and rejecting malformed tensor-family layouts
deterministically. It is not yet about text, vision, or audio runtime execution.

## Implementation Decisions

### Architecture Scope
- Accept `gguf.architecture=omniembed` through the existing model architecture registry rather than
  aliasing through an LLM decoder family.
- Keep the new support architecture-general: validate modality tensor families and shared embedding
  metadata without hard-coding TE-75M-only component names into the contract.
- Preserve the current milestone boundary that the first maintained `omniembed` slice is
  multimodal text-image-audio embeddings, not generation.

### Contract Shape
- Add a dedicated `src/emel/model/omniembed` family beside `llama`, `qwen3`, `lfm2`, and
  `gemma4`.
- Expose an internal `execution_contract` that groups the required tensor families
  (`text_encoder`, `text_projection`, `image_encoder`, `image_projection`, `audio_encoder`,
  `audio_projection`) and shared embedding metadata needed by later modality lanes.
- Keep this contract internal to `src/` and avoid widening any public C ABI or generator-facing
  runtime surface in Phase 48.

### Metadata Truth
- Load `omniembed.embed_dim`, `omniembed.image_encoder_dim`, `omniembed.audio_encoder_dim`, and
  `omniembed.matryoshka_dims` into repo-owned model metadata.
- Use existing clip-style metadata slots for modality presence and encoder/projection dimensions
  where that keeps the surface additive.
- Validate Matryoshka truncation metadata structurally now so later phases can trust it.

## Existing Code Insights

### Reusable Assets
- `src/emel/model/architecture/detail.cpp` is already the truthful architecture dispatch point for
  GGUF family selection.
- `src/emel/model/detail.cpp` already loads `general.architecture` first and resolves it through
  the architecture registry before vocab loading.
- `tests/model/loader/lifecycle_tests.cpp` already contains GGUF binding helpers and loader-level
  contract tests that make it the right place to pin `omniembed` acceptance behavior.

### Constraints
- `src/emel/model/data.cpp` is currently decoder-shaped for `llama`-style execution views, so
  Phase 48 must not pretend `omniembed` is a generator-compatible block contract.
- The user explicitly wants general architecture support rather than a TE-75M-only special case, so
  validation should focus on modality families and embedding contract truth, not one frozen encoder
  implementation detail.
- AGENTS requires state-machine structure approval before machine restructuring; this phase stays in
  the model-contract layer and does not change orchestration state machines.

## Specific Ideas

- Use family-prefix scans to validate modality tensor presence without baking TE component class
  names into the acceptance contract.
- Add a dedicated `is_omniembed_execution_architecture(...)` helper next to the existing `lfm2` and
  `gemma4` helpers.
- Keep failure deterministic: malformed or incomplete `omniembed` files must fail through the new
  architecture path rather than falling into any LLM builder.

## Deferred Ideas

- `text/encoders`, `vision/encoders`, and `audio/encoders` runtime implementation
- `embeddings/generator` orchestration
- generic `*/forward` reuse seams
- public embedding API or CLI commitments

---
*Phase: 48-omniembed-model-contract*
*Context gathered: 2026-04-14*
