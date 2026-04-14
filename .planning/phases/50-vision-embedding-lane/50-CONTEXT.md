---
phase: 50
slug: vision-embedding-lane
created: 2026-04-14
status: ready
---

# Phase 50 Context

## Phase Boundary

Phase 50 adds the maintained in-memory TE vision lane on top of the now-complete text lane. The
scope is intentionally narrow: accept one documented RGBA memory contract, run TE vision
preprocessing plus the TE vision encoder/projection path, and return a normalized shared-space
embedding. This phase does not widen into JPEG/PNG decode, generic image import, or public API
expansion.

## Implementation Decisions

### Runtime Surface
- Extend the existing `src/emel/embeddings/generator/` actor with an image request surface for the
  maintained slice rather than introducing public API churn in the same phase.
- Keep request modality routing explicit in the state machine; do not hide text/image choice in
  helper branching.
- Preserve the locked architectural direction (`vision/encoders/...`) as future cleanup, but keep
  this phase’s maintained runtime cut focused on truthful end-to-end behavior first.

### Vision Truth
- Drive the image lane from the pinned `TE-75M-q8_0.gguf` contract only.
- Read preprocessing dimensions and normalization stats from `model.meta.clip_vision_data` rather
  than hard-coding them.
- The tensor families in the maintained fixture show a convolutional/mobile vision tower
  (`image_encoder.blocks.*`, depthwise/pointwise conv, batch-norm families) followed by an
  explicit `image_projection.*` family. Phase 50 should implement that actual runtime path, not a
  placeholder CLIP/ViT assumption.

### Scope Guardrails
- Accept only the documented in-memory RGBA payload contract from
  `tests/embeddings/fixtures/te75m/README.md`.
- Keep image preprocessing deterministic and bounded; generic file decoding stays out of scope.
- Limit proof in this phase to maintained flow behavior, normalization, and invalid-payload
  rejection. Golden cross-modal comparison remains Phase 53 work.

## Existing Code Insights

### Reusable Assets
- `src/emel/model/omniembed/detail.cpp` already validates and exposes `image_encoder.*` and
  `image_projection.*` families through the execution contract.
- `src/emel/model/data.hpp` already carries the image metadata Phase 50 needs:
  `image_size`, `preproc_image_size`, `image_mean`, `image_std`, `projection_dim`, and related
  CLIP-style vision fields.
- `src/emel/embeddings/generator/` now provides the shared embedding-session actor, error/callback
  plumbing, and normalized publication contract proven by Phase 49.
- `tests/embeddings/fixtures/te75m/README.md` already defines the canonical maintained image
  anchor as a `32x32` RGBA red square synthesized directly in memory.

### Hard Constraints
- AGENTS still requires explicit SML routing, bounded actions, and no dispatch-time allocation.
- Phase 50 must not imply broad image support; it proves one truthful maintained memory contract.
- Quality gates must stay green even if failures surface outside the current lane.

## Specific Ideas

- Add an internal `embed_image` request to `embeddings/generator` with explicit success/error
  publication and invalid-request rejection.
- Add a narrow maintained image preprocessor that:
  - validates raw RGBA payload shape
  - converts the canonical image buffer into the model-required input layout
  - applies model-driven resize/normalize rules from `clip_vision_data`
- Implement the TE vision tower and projection path against the real `image_encoder.*` and
  `image_projection.*` tensors, then reuse the existing shared embedding publication logic.
- Add maintained tests that prove:
  - normalized `1280`-dimensional image embedding output
  - explicit invalid image payload rejection
  - compatibility with the canonical `red-square` RGBA anchor

## Deferred Ideas

- JPEG/PNG decoding
- arbitrary image sizes beyond the maintained contract
- public image embedding API
- golden-vector parity publication
- repo-wide `vision/encoders` extraction cleanup

## Primary Sources

- `tests/models/TE-75M-q8_0.gguf`
- `tests/embeddings/fixtures/te75m/README.md`
- `src/emel/model/omniembed/detail.cpp`
- `src/emel/model/data.hpp`
- `../../../lfg/lfg.cpp/src/vision/clip.cpp`

---
*Phase: 50-vision-embedding-lane*
*Context gathered: 2026-04-14*
