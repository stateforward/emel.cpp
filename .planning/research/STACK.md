# Stack Research

**Domain:** First maintained `augmem/TE-75M-GGUF` trimodal embedding slice for EMEL  
**Researched:** 2026-04-13  
**Confidence:** HIGH

## Milestone Recommendation

The first TE-75M milestone should pin one maintained GGUF artifact and keep the runtime surface
narrow:

- **Maintained fixture:** `TE-75M-q8_0.gguf`
- **Architecture:** `omniembed` (from the Hugging Face model API for `augmem/TE-75M-GGUF`)
- **Scope:** one synchronous feature-extraction slice for `text`, `image`, and `audio`
- **Proof style:** upstream-derived golden embeddings plus tiny cross-modal smoke checks

`TE-75M-q8_0.gguf` is the right first truth anchor because the upstream card describes it as the
minimal-quality-loss quant, and EMEL already has visible `q8_0` kernel/test surface. Starting with
`TE-75M-q5_0.gguf` would expand the first milestone into mixed-quant runtime work immediately.

## Reuse From The Existing Repo

| Existing Surface | Why It Helps |
|------------------|--------------|
| `src/emel/gguf/loader/` and `src/emel/model/loader/` | The repo already has the GGUF parse/bind/load orchestration needed to ingest a maintained TE fixture. |
| `src/emel/model/architecture/detail.*` | EMEL already resolves named execution architectures through an explicit registry; `omniembed` can follow the existing pattern used by `llama`, `qwen3`, `lfm2`, and `gemma4`. |
| `src/emel/text/conditioner/`, `src/emel/text/tokenizer/`, and `src/emel/text/encoders/` | These provide the closest actor-family reference for request shaping and explicit state-machine orchestration, even though the current `text/encoders` family is tokenizer-oriented rather than model-encoder-oriented. |
| `src/emel/kernel/events.hpp` and backend kernel SMs | The kernel layer already exposes 2D conv, depthwise conv, pooling, GELU, SiLU, norm, and matmul-style ops, so the first milestone is not blocked on inventing a new operator vocabulary. |
| `tests/models/README.md` and existing maintained-model workflow | EMEL already has a strong pattern for one exact maintained model artifact per supported slice. |

## Required Additions

| Area | Addition Needed | Why It Matters |
|------|------------------|----------------|
| Model family | `src/emel/model/omniembed/` architecture contract | The TE GGUF reports `gguf.architecture=\"omniembed\"`; truthful support needs an explicit family rather than aliasing to LLM paths. |
| Vision lane | New `vision` preprocessing and encoder actor family | There is no existing `src/emel/vision/` domain, but TE’s image path is part of the maintained claim. |
| Audio lane | New `audio` preprocessing and encoder actor family | There is no existing `src/emel/audio/` domain, and TE’s audio path is equally part of the maintained claim. |
| Shared embedding output | A repo-owned embedding session/output contract | TE returns normalized shared-space embeddings with Matryoshka truncation, not logits or generated tokens. |
| Proof fixtures | Upstream-derived golden text/image/audio embeddings | There is no obvious existing `llama.cpp` parity lane for `omniembed`, so the milestone needs its own deterministic reference proof. |
| Test assets | Tiny canonical text/image/audio triplet set | The shared-space claim should be proven with small fixed fixtures rather than benchmark datasets or ad hoc manual checks. |

## Open Stack Questions To Resolve In The Milestone

| Question | Recommended Resolution |
|----------|------------------------|
| Which TE quant should be maintained first? | `TE-75M-q8_0.gguf` only. Treat `TE-75M-q5_0.gguf` as future scope. |
| Should EMEL add generic image/audio file decoding in the same milestone? | No. Use one documented in-memory image contract and one documented in-memory audio contract for the first slice. |
| Should proof use the current generation parity toolchain? | No by default. Generate small upstream golden embeddings and compare EMEL against stored baselines in `emel_tests`. |
| Should TE support land behind a new public C API immediately? | No. Keep the first milestone on repo-internal/runtime seams first, then decide whether a public API milestone is warranted. |

## What To Avoid

| Avoid | Why |
|-------|-----|
| Starting with `TE-75M-q5_0.gguf` | It widens the first milestone into mixed-quant support before the architecture and modality lanes are even proven. |
| Generic multimedia file decoding | It adds codec, container, resampling, and metadata scope that is separate from proving one maintained embedding slice. |
| Treating TE as “just another text model” | TE is a feature-extraction model with modality-specific encoder lanes and a shared embedding contract, not a decoder. |
| Broad `omniembed` or generic multimodal support claims | The maintained truth should stay fixed to one TE-75M fixture and one narrow request surface. |
| Benchmark publication before proof | This repo’s core value requires proof first; TE should follow the same discipline even if the proof format differs from generation parity. |

## Recommended First-Slice Tooling

| Tool / Artifact | Recommendation |
|-----------------|----------------|
| `tests/models/README.md` | Add one maintained TE fixture entry with source URL, file name, license, size, and SHA256. |
| `tests/embedding/` | Add golden-vector tests and cross-modal smoke tests here or in similarly narrow new domains. |
| `scripts/quality_gates.sh` | Keep using the existing gate; TE proof should integrate into `emel_tests` rather than bypassing the normal workflow. |
| `.planning/research/*` | Keep the scope narrow to the maintained vertical slice, not a product roadmap for retrieval systems. |

## Sources

- `https://huggingface.co/augmem/TE-75M-GGUF`
- `https://huggingface.co/augmem/TE-75M`
- `https://huggingface.co/augmem/TE-75M-GGUF/raw/main/README.md`
- `https://huggingface.co/augmem/TE-75M/raw/main/README.md`
- `https://huggingface.co/api/models/augmem/TE-75M-GGUF`
- `https://huggingface.co/api/models/augmem/TE-75M`
- `src/emel/model/architecture/detail.cpp`
- `src/emel/kernel/events.hpp`
- `src/emel/text/encoders/sm.hpp`
- `tests/models/README.md`

---
*Stack research for: first maintained TE-75M GGUF trimodal embedding support in EMEL*  
*Researched: 2026-04-13*
