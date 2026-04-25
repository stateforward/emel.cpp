# Stack Research

**Domain:** ARM support for `openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf`
**Researched:** 2026-04-22
**Confidence:** MEDIUM

## Existing Repo Stack

- C++ runtime organized around Boost.SML actors and explicit event-driven orchestration.
- GGUF loading and model-family validation under `src/emel/model/**`.
- ARM kernel and tensor execution surfaces under `src/emel/kernel/**`.
- Existing audio support is embedding-specific through `src/emel/embeddings/generator` and
  `omniembed_efficientat_mn20_as`; there is no diarization or Sortformer runtime yet.
- Existing benchmark/parity tooling already separates EMEL and reference lanes for generation and
  embeddings. This pattern should carry into diarization proof.

## External Target Facts

- NVIDIA's upstream model card describes Streaming Sortformer as a diarization model that accepts
  mono 16 kHz audio and can return speaker segments or tensor outputs.
- The upstream output tensor is `T x S`, with `S = 4` speakers and one 0.08-second frame per row.
- Recommended streaming profiles are expressed in 80 ms frames, including `chunk_len`,
  `chunk_right_context`, `fifo_len`, `spkcache_update_period`, and `spkcache_len`.
- NVIDIA describes the architecture as Fast-Conformer/NEST pre-encoding plus Sortformer speaker
  cache and transformer processing.
- Public conversion/export discussion shows dynamic slicing and runtime-dependent indexing as an
  interoperability risk for static graph export. EMEL should not plan around ONNX as the runtime
  truth for this milestone.

## Recommended Stack Additions

```text
src/emel/model/sortformer/          # GGUF metadata, tensor-family validation, execution contract
src/emel/audio/diarization/         # diarization-owned request/result actor
src/emel/audio/diarization/sortformer/
                                     # maintained Sortformer orchestration components
src/emel/audio/frontend/sortformer/ # native PCM-to-feature/input tensor work if reused later
src/emel/kernel/**                  # kernel-owned numeric work needed by the runtime path
tools/bench/**                      # benchmark runner/wrapper additions only
tools/paritychecker/**              # reference-lane comparison additions only
snapshots/**                        # stored proof/publication only with explicit update consent
```

Exact directories can be adjusted during phase planning, but the namespace layout should keep
diarization distinct from embedding generation.

## Required Tools And Proof Surfaces

- GGUF fixture/provenance setup for the exact maintained openresearchtools artifact.
- Focused doctest coverage for loader/model acceptance and diarization actor behavior.
- Native reference baseline generation workflow, likely using the upstream NeMo model as the
  trusted reference lane, with no shared state between reference and EMEL lanes.
- One ARM benchmark publication path that records fixture identity, stream profile, audio duration,
  timing, and proof status.

## Stack Risks

- Treating Sortformer as another embedding lane would hide the product contract. The output is
  diarization probabilities and segments, not normalized embeddings.
- Treating ONNX or Python as the EMEL runtime path would fail the native ARM support goal.
- Adding media decode/resample/channel-mix stack during this milestone would widen the work beyond
  the runtime proof.

## Sources

- https://huggingface.co/openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf
- https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
- https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/
- https://github.com/NVIDIA-NeMo/NeMo/issues/15077
- `src/emel/model/architecture/detail.cpp`
- `src/emel/embeddings/generator/**`
- `tools/bench/**`

---
*Stack research for: ARM Sortformer diarization GGUF support in EMEL*
*Researched: 2026-04-22*
