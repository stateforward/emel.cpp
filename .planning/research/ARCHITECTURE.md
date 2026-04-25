# Architecture Research

**Domain:** ARM support for `openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf`
**Researched:** 2026-04-22
**Confidence:** MEDIUM

## Existing Architectural Footholds

- `src/emel/model/architecture/detail.cpp` already centralizes supported execution architectures.
  Sortformer support should enter through that model-family registry, not through a benchmark shim.
- `src/emel/model/data.hpp` already holds audio-shaped metadata for embedding models. Sortformer
  should add diarization-specific metadata deliberately instead of overloading `omniembed`.
- `src/emel/embeddings/generator` proves that audio payloads can be routed through explicit SML
  lanes, but its result contract is embeddings. Diarization should get its own actor/result shape.
- Existing `tools/bench` and `tools/paritychecker` work already enforce lane isolation. The same
  rule should govern any NeMo/reference comparison.

## Recommended Component Shape

```text
src/emel/model/sortformer/
  detail.hpp / detail.cpp      # hidden metadata binding helpers used more than once

src/emel/audio/diarization/
  any.hpp
  context.hpp
  events.hpp
  guards.hpp
  actions.hpp
  sm.hpp

src/emel/audio/diarization/sortformer/
  context.hpp
  events.hpp
  guards.hpp
  actions.hpp
  sm.hpp
  detail.hpp / detail.cpp      # shared non-routing numeric/binding helpers only

src/emel/kernel/**
  # ARM/native kernels required by Sortformer feature, conformer, transformer, cache, or head work
```

This keeps the top-level contract as diarization and the maintained implementation family as
Sortformer. If a future milestone adds a second diarization architecture, it can route through
`audio/diarization/any.hpp` without treating Sortformer as the whole domain.

## Recommended Data Flow

```text
maintained Sortformer GGUF fixture
    -> GGUF parse/bind/load
    -> model::sortformer execution contract
    -> audio/diarization request validates PCM/profile/output buffers
    -> Sortformer-owned frontend/input tensor preparation
    -> Sortformer encoder/cache/transformer/head execution
    -> T x 4 speaker probability matrix
    -> deterministic segment decoding
    -> callback/result or explicit error event
```

## SML Boundary Guidance

- Runtime choices must live in `guards.hpp` and `sm.hpp`: supported model contract, profile
  selection, cache readiness, output-capacity acceptance, and error outcomes.
- `actions.hpp` should execute an already-chosen path and remain bounded/non-blocking.
- `detail.hpp` and `detail.cpp` may contain shared non-routing numeric helpers, tensor binding, and
  data-plane iteration only.
- Do not store request pointers, output pointers, phase flags, frame counts, or temporary error
  status in context. Carry dispatch-local data through typed internal events.
- The maintained actor should expose an explicit unexpected-event behavior.

## Build Order Recommendation

1. Fixture and model contract: accept/reject truth before runtime.
2. Diarization actor shell: request/result/error surface and SML lifecycle.
3. Audio frontend/profile: mono PCM and input tensor preparation.
4. Native Sortformer execution: feature/pre-encoder, cache, transformer, and head path.
5. Output decoder: probabilities and segment records.
6. Proof and benchmark: lane-isolated reference comparison and ARM timing publication.

## Architectural Risks

| Risk | Why It Matters | Mitigation |
|------|----------------|------------|
| Reusing embeddings/generator for diarization | It would blur result contracts and make future diarization APIs awkward. | Add a diarization-owned actor and keep embeddings unchanged. |
| Hiding cache/profile choices in helpers | Violates SML behavior-selection rules. | Model choices as guards/transitions. |
| Using reference runtime in EMEL lane | Fails the native ARM support contract. | Confine NeMo/reference code to tools/paritychecker or tools/bench reference lanes. |
| Adding media ingestion during runtime bring-up | Expands determinism and test burden sharply. | Keep input as mono 16 kHz PCM. |
| Treating probability output and segment output as interchangeable | Probability matrices are the model output; segments are a post-processing contract. | Test both contracts explicitly. |

## Sources

- https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
- https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/
- `AGENTS.md`
- `src/emel/model/architecture/detail.cpp`
- `src/emel/model/data.hpp`
- `src/emel/embeddings/generator/sm.hpp`
- `tools/bench/**`
- `tools/paritychecker/**`

---
*Architecture research for: ARM Sortformer diarization GGUF support in EMEL*
*Researched: 2026-04-22*
