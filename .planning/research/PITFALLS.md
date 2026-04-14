# Pitfalls Research

**Domain:** First maintained `augmem/TE-75M-GGUF` trimodal embedding slice for EMEL  
**Researched:** 2026-04-13  
**Confidence:** HIGH

## Critical Pitfalls

### 1. Confusing Tokenizer Encoders With Model Encoders

**What goes wrong:** the milestone tries to force TE’s neural text lane into the existing
`src/emel/text/encoders/` tokenizer family.

**Why it is risky:** that family explicitly documents that it is distinct from multimodal model
encoders. Reusing it directly would blur responsibilities and make the architecture harder to
maintain.

**How to prevent it:** reuse text request shaping/tokenization where helpful, but keep TE neural
encoding in the new model/embedding work.

### 2. Picking `q5_0` For The First Maintained Slice

**What goes wrong:** the milestone starts with `TE-75M-q5_0.gguf` because it is slightly smaller.

**Why it is risky:** the first slice already needs a new architecture family and three modality
lanes. Adding mixed-quant support at the same time makes the milestone much wider immediately.

**How to prevent it:** pin `TE-75M-q8_0.gguf` first, then treat `q5_0` as future scope.

### 3. Hiding Modality Choice In Actions Or Helpers

**What goes wrong:** the implementation chooses `text` vs `image` vs `audio` inside helper code or
action branching.

**Why it is risky:** that directly conflicts with the repo’s SML rules, which require runtime
behavior choice to remain explicit in guards, states, and transitions.

**How to prevent it:** model the modality lane as an explicit dispatch/decision graph.

### 4. Turning The Milestone Into A Media-Decoding Project

**What goes wrong:** the first slice tries to solve JPEG/PNG/WAV/MP3 decode, resampling, channel
mixing, and metadata handling together with TE inference.

**Why it is risky:** those are real product features, but they are not required to prove one
maintained embedding slice and they widen determinism and testing scope sharply.

**How to prevent it:** keep first inputs as narrow in-memory contracts.

### 5. Lacking A Trustworthy Reference Proof

**What goes wrong:** the milestone lands with flow tests only, but no stable upstream output
baseline.

**Why it is risky:** TE is a new architecture family and there is no obvious current generation
parity lane for it. Without reference outputs, “supported” becomes hard to verify.

**How to prevent it:** store tiny upstream-generated golden embeddings and compare EMEL against
them in `emel_tests`.

### 6. Forgetting The Shared Embedding Contract

**What goes wrong:** per-modality lanes land, but they do not converge on one normalized output
contract or one Matryoshka truncation contract.

**Why it is risky:** the product claim is not “three unrelated encoders”; it is one shared-space
embedding model.

**How to prevent it:** make shared output normalization, supported dimensions, and deterministic
dimension rejection explicit milestone requirements.

### 7. Widening The Public API Too Early

**What goes wrong:** the milestone tries to settle a public C ABI or broad CLI for multimedia
embedding requests before the maintained slice is proven.

**Why it is risky:** that would lock in surface area before the architecture, request contracts,
and proof story are stable.

**How to prevent it:** keep the first milestone focused on maintained internal/runtime truth, then
decide whether a public API milestone is justified.

## Pitfall To Phase Mapping

| Pitfall | Phase To Address |
|---------|------------------|
| Tokenizer/model-encoder confusion | Architecture and text-lane phases |
| `q5_0` first-slice scope creep | Fixture truth-anchor phase |
| Hidden modality routing | Shared embedding session phase |
| Media-decode scope creep | Vision and audio lane phases |
| Missing reference proof | Proof and regression phase |
| Missing shared-space contract | Shared embedding session phase |
| Premature public API | Out-of-scope guardrails across the milestone |

## Warning Signs

- docs say “TE support” without naming one exact maintained file
- implementation says `omniembed` is supported before the model contract exists
- request handling accepts file paths/codecs instead of narrow in-memory payloads
- text/image/audio lanes return different result shapes or different truncation rules
- proof relies on manual inspection rather than stored upstream baselines
- roadmap starts discussing vector indexes, ANN search, or public APIs before the core slice works

## Sources

- `https://huggingface.co/augmem/TE-75M-GGUF`
- `https://huggingface.co/augmem/TE-75M`
- `https://huggingface.co/api/models/augmem/TE-75M-GGUF`
- `src/emel/text/encoders/sm.hpp`
- `AGENTS.md`

---
*Pitfalls research for: first maintained TE-75M GGUF trimodal embedding support in EMEL*  
*Researched: 2026-04-13*
