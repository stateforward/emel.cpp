# Requirements: EMEL

**Defined:** 2026-04-13  
**Reopened:** 2026-04-14 after the `v1.11` milestone audit found closeout traceability and
runtime-integration gaps.

**Core Value:** Prove real end-to-end behavior with explicit SML orchestration and parity-oriented
verification before widening API surface or model scope.

## v1 Requirements

### Fixture Identity

- [x] **FIX-01**: Maintainer can locate one official maintained TE fixture at
  `tests/models/TE-75M-q8_0.gguf` with recorded provenance, download URL, license, size, and
  checksum.
- [x] **FIX-02**: The maintained TE workflow accepts only the approved TE fixture variants
  `tests/models/TE-75M-q8_0.gguf` and `tests/models/TE-75M-q5_0.gguf`; other sibling artifacts
  remain unapproved and are not part of the maintained `v1.11` slice.

### Model Contract

- [x] **MOD-01**: EMEL recognizes `gguf.architecture=omniembed` and validates the required TE
  text, image, audio, and projection tensor families explicitly.
- [x] **MOD-02**: EMEL builds a TE execution contract for `omniembed` without aliasing the model
  to an existing LLM generation path.

### Text Embedding

- [x] **TXT-01**: Operator can submit text input on the maintained TE path and receive a normalized
  1280-dimensional embedding.
- [x] **TXT-02**: Operator can request supported Matryoshka truncation for text embeddings at
  `768`, `512`, `256`, or `128` dimensions, with renormalization after truncation.

### Vision Embedding

- [x] **VIS-01**: Operator can submit one documented in-memory image payload on the maintained TE
  path and receive a normalized 1280-dimensional embedding.
- [x] **VIS-02**: Invalid image payload shape or preprocessing contract is rejected explicitly on
  the maintained TE path.

### Audio Embedding

- [x] **AUD-01**: Operator can submit one documented in-memory audio payload on the maintained TE
  path and receive a normalized 1280-dimensional embedding.
- [x] **AUD-02**: Invalid audio payload shape or preprocessing contract is rejected explicitly on
  the maintained TE path.

### Shared Embedding Contract

- [x] **EMB-01**: Text, image, and audio lanes all return one consistent embedding result contract
  with shared normalization and explicit modality-aware errors.
- [x] **EMB-02**: The first maintained TE slice stays synchronous, bounded, and limited to one
  modality per request.

### Proof

- [x] **PRF-01**: Maintained tests compare EMEL outputs against stored upstream TE golden
  embeddings for canonical text, image, and audio fixtures.
- [x] **PRF-02**: Maintained smoke tests prove the shared-space contract on a tiny canonical
  cross-modal triplet set.

## v2 Requirements

### Follow-On Scope

- **IO-01**: EMEL accepts common image/audio file formats and performs required decode or resample
  steps before TE inference.
- **API-01**: EMEL exposes a stable public C or CLI embedding API for the TE slice.
- **BCH-01**: EMEL supports small bounded same-modality embedding batches on the TE path.

## Out of Scope

| Feature | Reason |
|---------|--------|
| TE quant siblings beyond the approved `TE-75M-q8_0.gguf` and `TE-75M-q5_0.gguf` fixtures | Keep `v1.11` pinned to the two maintained TE slices actually proved in this repo |
| Generic image/audio file decoding and transcoding | Separate product surface from proving one maintained embedding lane |
| Vector search, ANN index management, reranking, or retrieval serving | TE enables retrieval, but this milestone is only about embedding extraction |
| Multimodal generation, captioning, transcription, or chat claims | TE-75M is a feature-extraction model, not the next maintained generation slice |
| Broad `omniembed` or arbitrary embedding-model support | The milestone is fixed to one maintained TE-75M slice |
| Public multimedia C ABI or CLI commitments | Prove the runtime first, then decide whether a public API milestone is justified |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FIX-01 | Phase 47 | Complete |
| FIX-02 | Phase 60 | Complete |
| MOD-01 | Phase 48 | Complete |
| MOD-02 | Phase 54 | Complete |
| TXT-01 | Phase 55 | Complete |
| TXT-02 | Phase 55 | Complete |
| VIS-01 | Phase 55 | Complete |
| VIS-02 | Phase 55 | Complete |
| AUD-01 | Phase 55 | Complete |
| AUD-02 | Phase 55 | Complete |
| EMB-01 | Phase 54 | Complete |
| EMB-02 | Phase 55 | Complete |
| PRF-01 | Phase 56 | Complete |
| PRF-02 | Phase 56 | Complete |

**Coverage:**
- v1 requirements: 14 total
- Mapped to phases: 14
- Complete: 14
- Pending: 0
- Unmapped: 0

---
*Requirements defined: 2026-04-13*  
*Reopened: 2026-04-14 for v1.11 audit gap closure*
