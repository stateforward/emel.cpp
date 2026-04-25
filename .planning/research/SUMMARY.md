# Research Summary

**Milestone:** `v1.15 ARM Sortformer Diarization GGUF Slice`
**Summarized:** 2026-04-22

## Recommended Scope

Build one truthful maintained Sortformer diarization vertical slice:

- pin `openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf` as the maintained GGUF target
- add explicit Sortformer model-family acceptance and execution-contract validation
- add a diarization-owned actor for mono 16 kHz PCM requests
- run the maintained Sortformer path natively in EMEL-owned `src/` code on ARM
- emit deterministic `T x 4` probabilities and bounded four-speaker segment records
- prove behavior against a lane-isolated trusted reference baseline
- publish one maintained ARM benchmark with fixture/profile/proof metadata

## Stack Additions

- `src/emel/model/sortformer/` for GGUF/model-family contract and tensor-family validation
- `src/emel/audio/diarization/` for the diarization request/result/error actor
- `src/emel/audio/diarization/sortformer/` for the maintained Sortformer implementation family
- `src/emel/kernel/**` additions for Sortformer numeric work that belongs in kernel ownership
- `tools/paritychecker/**` or `tools/bench/**` reference-lane additions for proof and benchmark

## Table Stakes

- one exact maintained GGUF fixture contract with provenance/checksum
- explicit loader rejection for incompatible or incomplete Sortformer GGUF files
- one mono `float32` PCM, 16,000 Hz input contract
- native frontend/input tensor preparation
- native ARM Sortformer execution without external compute fallback
- deterministic `T x 4` speaker probability output with 0.08-second frame semantics
- stable `speaker_0` through `speaker_3` segment output
- lane-isolated reference proof and ARM benchmark publication

## Watch Out For

- do not route this through embeddings/generator just because existing audio embedding support
  exists
- do not use Python, NeMo, ONNX, llama.cpp, or ggml as the EMEL compute path
- do not add media decode, resampling, live microphone capture, or service streaming in v1
- do not hide streaming profile or cache-readiness routing in actions/detail helpers
- do not call the milestone maintained with segment-only proof and no probability-matrix truth
- do not share model/cache/tensor/output objects between EMEL and reference lanes

## Sources

- `.planning/research/STACK.md`
- `.planning/research/FEATURES.md`
- `.planning/research/ARCHITECTURE.md`
- `.planning/research/PITFALLS.md`
- https://huggingface.co/openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf
- https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
- https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/

---
*Research summarized: 2026-04-22*
