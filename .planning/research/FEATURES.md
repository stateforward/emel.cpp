# Feature Research

**Domain:** First maintained `augmem/TE-75M-GGUF` trimodal embedding slice for EMEL
**Researched:** 2026-04-13
**Confidence:** MEDIUM

## Milestone Recommendation

The first TE-75M milestone should establish one truthful maintained embedding vertical slice, not a
general multimodal platform. EMEL is currently generation-oriented, its existing `text/encoders`
cluster explicitly excludes multimodal model encoders, and there is no existing `vision`, `audio`,
or embedding-domain actor family in `src/emel/`.

Recommendation: pin **one** official GGUF file first, and make it
`TE-75M-q8_0.gguf`. This is an inference from two facts: Hugging Face positions `q8_0` as the
minimal-quality-loss quant, and EMEL already has visible `q8_0` kernel/test surface while `q5_0`
support is not visibly implemented beyond dtype/loader recognition. Starting with `q5_0` would
turn the first maintained slice into a broader quant-runtime milestone immediately.

## Table Stakes

These are the operator-facing feature categories that belong in the first maintained TE-75M
milestone.

| Category | Why It Belongs In v1 | Proof / Testing Implication | Likely Dependencies |
|----------|-----------------------|-----------------------------|---------------------|
| **Pinned maintained TE fixture** | EMEL’s maintained model pattern starts with one exact official artifact under `tests/models/`. TE-75M support is not truthful until the repo names one canonical file, URL, and checksum. | Add one `tests/models/README.md` entry and loader/integration coverage that rejects drifted or wrong fixtures. | Official Hugging Face artifact, stable local path, checksum, documented provenance. |
| **Explicit `omniembed` model acceptance** | Hugging Face API reports `gguf.architecture=\"omniembed\"`; EMEL does not currently support that family. Operators need truthful accept/reject behavior, not silent aliasing to LLM paths. | Loader tests should verify architecture detection, required tensor families, projection-head presence, and explicit rejection of unsupported TE siblings or malformed GGUFs. | New model-family loader/validation rules; TE-specific execution views; fixture metadata. |
| **Text embedding request path** | First maintained support must let operators submit text and receive a TE shared-space embedding. Reusing only the existing generator/text stack is insufficient because TE is an embedding model, not a decoder. | New machine flow tests and golden-vector tests for text inputs; explicit unexpected-event/error-path coverage. | Narrow text request contract, TE text encoder actor, TE text projection path, shared embedding output contract. |
| **Image embedding request path** | TE-75M is marketed as trimodal. If image input is missing, the maintained claim is incomplete. The first slice should accept one documented in-memory image contract rather than broad file-format decoding. | Flow tests for valid/invalid image requests plus golden-vector tests on a tiny maintained image set. | Vision preprocessing contract, vision encoder state machine, image projection path, maintained image fixtures. |
| **Audio embedding request path** | The milestone needs real audio support, not text-image only support under a trimodal name. The first slice should accept one documented in-memory audio contract and keep broad media ingestion out of scope. | Flow tests for valid/invalid audio requests plus golden-vector tests on a tiny maintained audio set. | Audio preprocessing contract, audio encoder state machine, audio projection path, maintained audio fixtures. |
| **Shared embedding output contract** | Operators need one consistent result shape regardless of modality: L2-normalized shared-space vectors plus the advertised Matryoshka truncation options. This is core product behavior, not optional polish. | Verify full 1280-d output, valid truncation to `768/512/256/128`, renormalization after truncation, and explicit rejection of unsupported dimensions. | Per-modality encode paths, projection-head correctness, output buffer contract. |
| **Deterministic error and capability reporting** | First maintained support must tell operators what is unsupported: wrong modality payload shape, unsupported dimension, wrong model file, and unsupported quant. Silent fallback would be misleading. | Negative tests should cover bad requests, unsupported quant/file choices, and unexpected-event handling. | Stable request schema, explicit error codes/events, narrow supported surface. |
| **Maintained proof surface for trimodal embeddings** | EMEL cannot call the milestone “maintained” on ad hoc manual runs. Operators need repeatable evidence that the three modality lanes land in the same usable space. | Add golden embeddings from the official upstream model lane, cross-modal smoke checks on tiny canonical triplets, and regression tests that run in `emel_tests`. | Fixture provenance, canonical text/image/audio examples, golden-baseline generation workflow. |

## Differentiators That Matter

These are valuable follow-ons that materially improve operator trust and usability, but they are
not required to call the first TE-75M slice complete.

| Feature | Value | Proof / Testing Implication | Depends On |
|---------|-------|-----------------------------|------------|
| **Tiny cross-modal retrieval smoke tool** | Makes the shared-space claim auditable by showing a known text/image/audio triplet ranks together without requiring a vector database. | Add a small deterministic smoke corpus and nearest-neighbor expectation tests. | All three modality lanes, golden baselines, shared embedding contract. |
| **Small-batch encode support** | Improves real operator throughput for ingestion jobs without forcing a full mixed-modality scheduler in v1. | Add bounded-work tests and batch-shape coverage; keep the first version same-modality only. | Stable single-item encode flow, output buffer contract, capacity/error rules. |
| **Published dimension-vs-cost guidance** | TE-75M’s Matryoshka surface is more useful when operators can see when `1280` vs `256` is worth using. | Add benchmark/docs publication for the maintained fixture and dimensions only after correctness is stable. | Shared embedding contract, maintained benchmark harness. |
| **Second official quant after launch** | Once the q8_0 truth anchor is stable, supporting `q5_0` becomes a real product expansion rather than speculative breadth. | Needs separate runtime proof and explicit negative/positive quant tests. | Stable q8_0 launch, additional quant-runtime implementation, new maintained fixture decision. |

## Anti-Features And Explicit Deferments

These should be kept out of the first maintained milestone.

| Deferred Surface | Why Defer | What To Do Instead In v1 |
|------------------|-----------|---------------------------|
| **Broad quant-matrix support (`q5_0` plus `q8_0`)** | The first milestone should prove one truthful slice. `q5_0` is a different runtime surface and is not visibly covered by today’s kernel implementation. | Launch with `TE-75M-q8_0.gguf` only and reject other quants explicitly. |
| **Generic media file decoding/transcoding** | JPEG/PNG/WAV/MP3 decode, resampling, channel mixing, and metadata handling are a separate product surface from inference. They also widen determinism and test scope immediately. | Accept one narrow in-memory image contract and one narrow in-memory audio contract. |
| **Vector index / ANN search / reranking inside EMEL** | TE-75M enables retrieval, but index management is not the inference engine’s first maintained obligation. | Stop at embedding extraction plus optional smoke-level similarity proof. |
| **Multimodal chat, captioning, transcription, or generation claims** | TE-75M is a feature-extraction model, and EMEL’s current maintained story is generation-oriented. Mixing those product surfaces would muddy the milestone. | Keep the milestone strictly about embeddings. |
| **Raw encoder hidden-state exposure** | Operators primarily need final shared-space embeddings. Exposing intermediate text/image/audio features broadens API and test surface with little first-milestone value. | Return final normalized embeddings only. |
| **Mixed-modality scheduling and streaming** | Streaming ingestion and heterogeneous batch planning are useful later, but they are not required to prove one synchronous trimodal slice. | Keep first requests synchronous and bounded, one modality per request. |
| **Public C ABI / CLI expansion as part of the first slice** | EMEL’s public C surface is currently minimal. A stable public multimedia API is a separate commitment from proving internal maintained support. | Land the milestone first on the existing repo/testing/tool seams, then decide if a public API milestone is warranted. |
| **Performance publication before correctness proof** | Benchmark numbers without trustworthy embeddings would create a false “supported” story. | Add docs/bench publication only after golden-vector and smoke-proof coverage is green. |

## Feature Dependencies

```text
[Pinned TE-75M-q8_0 fixture]
    └──requires──> [Provenance + checksum + stable tests/models path]

[Explicit omniembed model acceptance]
    └──requires──> [Pinned TE-75M-q8_0 fixture]

[Text embedding path]
    ├──requires──> [Explicit omniembed model acceptance]
    └──requires──> [TE text encoder + text projection]

[Image embedding path]
    ├──requires──> [Explicit omniembed model acceptance]
    └──requires──> [Vision preprocessing contract + vision encoder + image projection]

[Audio embedding path]
    ├──requires──> [Explicit omniembed model acceptance]
    └──requires──> [Audio preprocessing contract + audio encoder + audio projection]

[Shared embedding output contract]
    ├──requires──> [Text embedding path]
    ├──requires──> [Image embedding path]
    └──requires──> [Audio embedding path]

[Maintained trimodal proof surface]
    ├──requires──> [Shared embedding output contract]
    └──requires──> [Golden baselines from official upstream TE model]

[Cross-modal retrieval smoke tool]
    └──requires──> [Maintained trimodal proof surface]

[Second quant support]
    ──conflicts-with──> [Narrow first-slice q8_0 scope]

[Generic media file decode]
    ──conflicts-with──> [Narrow deterministic in-memory modality contracts]
```

## MVP Recommendation

Prioritize these categories for the first maintained TE milestone:

1. **Pinned q8_0 truth anchor**
   - `TE-75M-q8_0.gguf` only
   - documented provenance and checksum
   - explicit rejection of other TE files for now

2. **Three synchronous modality lanes**
   - text -> shared embedding
   - image -> shared embedding
   - audio -> shared embedding
   - one narrow request contract per modality, in memory only

3. **One shared output contract**
   - normalized embedding output
   - supported truncation to `1280/768/512/256/128`
   - deterministic errors for unsupported dimensions and invalid payloads

4. **Maintained proof, not just bring-up**
   - golden-vector baselines from the official TE upstream lane
   - tiny cross-modal smoke expectations
   - regression coverage in `emel_tests`

## Testing And Proof Notes

- **Do not assume the existing `llama.cpp` parity lane applies.**
  I did not find `omniembed` in EMEL, and I did not find an obvious `omniembed` code-search hit in
  `ggml-org/llama.cpp`. Treat TE proof as needing a new baseline source unless later research proves
  a reference runtime exists.
- **Golden vectors are table stakes here.**
  The cleanest proof surface is to generate a tiny canonical set of text/image/audio embeddings from
  the official `augmem/TE-75M` upstream lane and check EMEL against those stored baselines.
- **Cross-modal smoke should be tiny and deterministic.**
  Use a few canonical triplets, not a benchmark dataset import. The goal is proof of shared-space
  behavior, not productized retrieval evaluation.
- **Test naming should mirror new domains.**
  Likely domains are `tests/embedding/`, `tests/vision/`, and `tests/audio/`, with one file per
  machine or behavior, aligned with `AGENTS.md`.
- **Quality gates still apply.**
  The milestone should fit the existing `emel_tests` and `scripts/quality_gates.sh` workflow even
  if TE-specific proof replaces the usual generation parity pattern.

## Requirement-Shaped Categories

These are the categories most ready to become requirement sections or REQ IDs.

- **Fixture Identity And Provenance**
- **TE Model Acceptance And Validation**
- **Text Embedding Request Surface**
- **Image Embedding Request Surface**
- **Audio Embedding Request Surface**
- **Shared Embedding Output And Truncation**
- **Capability Rejection And Error Reporting**
- **Golden Baselines And Trimodal Regression Proof**

## Sources

### Official Sources

- https://huggingface.co/augmem/TE-75M-GGUF
- https://huggingface.co/augmem/TE-75M
- https://huggingface.co/augmem/TE-75M-GGUF/raw/main/README.md
- https://huggingface.co/augmem/TE-75M/raw/main/README.md
- https://huggingface.co/api/models/augmem/TE-75M-GGUF
- https://huggingface.co/api/models/augmem/TE-75M

### Repo Sources

- `.planning/PROJECT.md`
- `.planning/codebase/ARCHITECTURE.md`
- `.planning/codebase/INTEGRATIONS.md`
- `.planning/codebase/TESTING.md`
- `src/emel/text/encoders/sm.hpp`
- `src/emel/text/encoders/any.hpp`
- `tests/models/README.md`
- `src/emel/kernel/events.hpp`
- `tests/kernel/lifecycle_tests.cpp`
- `src/emel/kernel/aarch64/sm.hpp`

---
*Feature research for: first maintained TE-75M GGUF trimodal embedding support in EMEL*
*Researched: 2026-04-13*
