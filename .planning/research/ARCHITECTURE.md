# Architecture Research

**Domain:** First maintained `augmem/TE-75M-GGUF` trimodal embedding slice for EMEL  
**Researched:** 2026-04-13  
**Confidence:** HIGH

## Existing Architectural Footholds

The current repo is not multimodal yet, but it has several important footholds:

- `src/emel/model/architecture/detail.*` already uses an explicit architecture registry, so
  `omniembed` can land as a first-class model family.
- `src/emel/model/data.hpp` already carries some vision/audio-shaped metadata fields
  (`clip_vision_data`, `clip_audio_data`), which suggests the model-data layer can hold
  multimodal contracts without inventing a new top-level pattern.
- `src/emel/text/encoders/` shows the existing “one domain, one `any` router, multiple specific
  implementations” actor-family pattern, but it is explicitly a tokenizer encoder cluster, not a
  model encoder cluster.
- `src/emel/kernel/events.hpp` already exposes the operator classes a TE-style lane is likely to
  need: conv, depthwise conv, pooling, unary activations, norms, and matmul-family dispatch.

## Recommended Component Layout

The safest layout is to keep TE support explicit and additive while freeing the `text` domain for
future embedding-capable text producers:

```text
src/emel/model/omniembed/        # GGUF/model-family contract and execution bindings
src/emel/text/tokenizers/        # tokenizer families moved down from the current text/encoders tree
src/emel/text/encoders/          # text embedding producers
src/emel/vision/preprocessor/    # narrow image input shaping for the maintained slice
src/emel/vision/encoders/        # vision model-encoder actor family
src/emel/audio/preprocessor/     # narrow audio input shaping for the maintained slice
src/emel/audio/encoders/         # audio model-encoder actor family
src/emel/embeddings/generator/   # shared embedding-session orchestration
```

### Text-Lane Note

The locked direction is:

- move the current tokenizer-family implementation tree under `src/emel/text/tokenizers/`
- free `src/emel/text/encoders/` for text-to-embedding producers
- keep `src/emel/embeddings/generator/` as the milestone embedding orchestrator
- keep the existing `src/emel/generator/` where it is for this milestone

This keeps text tokenization and text embedding production as separate responsibilities while
leaving room for future embedding dispatch from generative-capable text models.

### Forward-Seam Rule

`forward` is a **reuse seam**, not a modality label.

- Every modality-specific model will have some forward computation internally.
- A public `*/forward/` domain is only justified when one modality has multiple higher-level
  contracts reusing the same hidden-state execution path.
- For `v1.11`, `text`, `vision`, and `audio` can keep their forward internals inside encoder
  families; a public `*/forward/` domain is future work, not a milestone requirement.
- If future text, vision, or audio work needs both generative and embedding contracts on the same
  hidden-state path, a domain-local `*/forward/` split becomes justified at that time.

## Recommended Data Flow

```text
maintained TE fixture (`TE-75M-q8_0.gguf`)
    ↓
GGUF parse/bind/load
    ↓
`model::omniembed` execution contract
    ↓
request chooses one modality lane
    ↓
modality-specific preprocessing
    ↓
    modality-specific encoder state machine
    ↓
    modality-specific projection head
    ↓
shared embedding normalization + optional Matryoshka truncation
    ↓
embedding result + deterministic error reporting
```

## Acceptance Boundary

The first milestone should stop at one truthful maintained embedding vertical slice:

- one pinned model file: `TE-75M-q8_0.gguf`
- one modality per request
- synchronous same-RTC execution
- final shared-space embeddings only
- one narrow input contract per modality
- reference proof via stored upstream golden baselines

It should **not** claim:

- generic multimedia decode/import
- generic multimodal scheduling
- vector-index or search service behavior
- public product API stability
- broad `omniembed` family support beyond TE-75M

## Build Order Recommendation

### 1. Truth Anchor And Reference Inputs

Pin the maintained TE fixture and decide the canonical text/image/audio fixtures used for proof.
Without this, later phases will drift on what “supported TE-75M” actually means.

### 2. `omniembed` Model Contract

Add explicit architecture acceptance, required tensor-family validation, and TE-specific execution
bindings. This keeps the repo honest: unsupported TE files should fail explicitly before the new
runtime exists.

### 3. Text Embedding Lane

Bring up the lowest-friction modality first by reusing the repo’s existing text request pipeline
where appropriate, then adding the text embedding encoder/projection runtime needed to produce
shared-space embeddings.

### 4. Vision Embedding Lane

Add a narrow in-memory image preprocessing contract and a dedicated vision encoder actor family.
Do not mix this with generic JPEG/PNG file ingestion in the same phase.

### 5. Audio Embedding Lane

Add a narrow in-memory audio preprocessing contract and a dedicated audio encoder actor family.
Keep broad decode/resample/channel-mix work out of scope.

### 6. Shared Embedding Session

Unify modality results behind one embedding contract:

- normalized output
- supported truncation to `1280/768/512/256/128`
- explicit dimension and payload validation
- one modality per request

For this milestone, that contract is owned by `src/emel/embeddings/generator/`, which injects the
text, vision, and audio encoder families and publishes one embedding-facing request/result surface.

### 7. Proof And Regression

Add golden-vector comparisons and tiny cross-modal smoke checks. Because the obvious current
generation-parity lane does not apply cleanly to `omniembed`, this proof phase is first-class
milestone work, not a follow-up.

## Architectural Risks

| Risk | Why It Matters | Mitigation |
|------|----------------|------------|
| Reusing tokenizer encoders as model encoders | It would blur two different responsibilities and make later maintenance harder. | Move tokenizer families under `text/tokenizers` and reserve `text/encoders` for embedding producers. |
| Forcing `*/forward` domains too early | It would add naming and file churn before any modality has proven hidden-state reuse across multiple terminal contracts. | Introduce `*/forward` only when a modality actually needs a shared hidden-state seam for more than one top-level contract. |
| Letting modality routing hide in helpers | That would violate the repo’s SML rules and make behavior hard to audit. | Keep modality choice explicit in guards/states/transitions. |
| Overloading the first slice with file decode or search features | It would turn a support milestone into a product-platform milestone. | Keep inputs in memory and stop at embeddings. |
| Missing a reference proof surface | Without a stable baseline, the milestone would widen the API without trustworthy behavior evidence. | Generate and store tiny upstream golden baselines before calling the slice maintained. |

## Sources

- `https://huggingface.co/augmem/TE-75M-GGUF/raw/main/README.md`
- `https://huggingface.co/augmem/TE-75M/raw/main/README.md`
- `https://huggingface.co/api/models/augmem/TE-75M-GGUF`
- `src/emel/model/architecture/detail.cpp`
- `src/emel/model/data.hpp`
- `src/emel/text/encoders/sm.hpp`
- `src/emel/text/encoders/any.hpp`
- `src/emel/text/tokenizer/sm.hpp`
- `src/emel/kernel/events.hpp`

---
*Architecture research for: first maintained TE-75M GGUF trimodal embedding support in EMEL*  
*Researched: 2026-04-13*
