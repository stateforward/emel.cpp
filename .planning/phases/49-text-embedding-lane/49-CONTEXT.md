---
phase: 49
slug: text-embedding-lane
created: 2026-04-14
status: complete
---

# Phase 49 Context

## Phase Boundary

Phase 49 brings up the first maintained runtime lane on top of the new `omniembed` model
contract: text in, TE text embedding out. This phase is about running the real TE text tower,
publishing a normalized shared-space embedding, and supporting the maintained Matryoshka
truncation contract for text requests. It is not yet the full multimodal embedding session and it
does not widen the public C ABI.

## Implementation Decisions

### Runtime Surface
- Introduce `src/emel/embeddings/generator/` as the first embedding-facing orchestration actor for
  the milestone.
- Keep Phase 49 request scope text-only even though the enclosing model contract is trimodal.
- Reuse the existing text conditioner/tokenizer request-shaping path instead of inventing a second
  tokenizer stack for the maintained slice.

### Text Tower Truth
- Treat the TE text lane as `LEAF-IR` plus the TE text projection head, not as a generative decode
  path.
- Run the base text encoder according to the upstream model contract:
  `BertModel -> mean pooling -> dense(384 -> 768) -> normalize`.
- Then run the TE text projection head according to the upstream TE model card:
  `Linear(768 -> 1920) -> GELU -> LayerNorm -> residual Linear(1920 -> 1920) -> GELU -> LayerNorm
  -> add residual -> Linear(1920 -> 1280) -> L2 normalize`.

### Scope Guardrails
- Keep the phase internal and maintained: real TE fixture, real text prompts, deterministic
  text-to-embedding output contract.
- Defer the public embedding API, batching commitments, image/audio runtime, and cross-modal
  orchestration to later phases.
- Do not force the repo-wide `text/tokenizers` migration into the same phase as the first runtime
  lane. Phase 49 must avoid conflating tokenizer encoders with TE model execution, but it does not
  need to churn the existing tokenizer subtree before the text lane is proven.

## Existing Code Insights

### Reusable Assets
- `src/emel/text/conditioner/` already owns the maintained text formatting and tokenization flow
  used by generator initialization and request shaping.
- `src/emel/text/tokenizer/detail.hpp` already maps BERT tokenizer metadata onto the repo-owned WPM
  path with the correct special-token defaults (`[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`).
- `src/emel/kernel/detail.hpp` already contains the q8_0 and f16/f32 operand helpers needed for
  TE text matmuls, plus dense softmax primitives that can be reused inside the text tower.
- `tests/models/TE-75M-q8_0.gguf` and `tests/embeddings/fixtures/te75m/*.txt` already lock the
  maintained local fixture and canonical text anchors for proof.

### Hard Constraints
- AGENTS requires explicit state-machine orchestration, branch-free actions, and no allocation
  during dispatch. Any large buffers needed by the text lane must therefore be reserved once before
  the first request.
- The maintained GGUF proves the TE text encoder is quantized (`q8_0`) while the projection head is
  split across `q8_0`, `f32`, and `f16` tensors. The runtime must respect those native operand
  classes instead of collapsing everything to a tool-only fallback lane.
- Phase 49 must not claim generic `omniembed` interoperability or public API stability. It only
  proves the repo-owned maintained TE text lane.

## Specific Ideas

- Mirror the existing generator pattern with an `initialize` event that binds the text conditioner
  once, then a text embedding request event that formats, tokenizes, runs the TE text tower, applies
  optional Matryoshka truncation, and publishes one embedding result.
- Keep the text tower math in the owning embedding component because it is only used once in this
  phase; do not invent a public `text/forward` or `text/encoders` runtime family yet.
- Add maintained-fixture tests that verify:
  - text request accepted on `TE-75M-q8_0.gguf`
  - `1280`-dim output length
  - L2 normalization
  - supported truncation to `768/512/256/128` with renormalization
  - explicit rejection of unsupported truncate dimensions

## Deferred Ideas

- public embedding C API / CLI
- image and audio runtime lanes
- shared multimodal embedding request/result surface
- generic batching
- repo-wide tokenizer namespace migration
- phase-53 golden-vector proof and regression corpus

## Primary Sources

- `https://huggingface.co/augmem/TE-75M`
- `https://huggingface.co/augmem/TE-75M-GGUF`
- `https://huggingface.co/MongoDB/mdbr-leaf-ir`
- `tests/models/TE-75M-q8_0.gguf`
- `tests/embeddings/fixtures/te75m/README.md`

---
*Phase: 49-text-embedding-lane*
*Context gathered: 2026-04-14*
