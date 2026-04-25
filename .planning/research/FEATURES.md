# Feature Research

**Domain:** ARM support for `openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf`
**Researched:** 2026-04-22
**Confidence:** MEDIUM

## Milestone Recommendation

The first Sortformer milestone should establish one truthful maintained diarization vertical slice,
not a general speech platform. The target is a GGUF conversion of NVIDIA Streaming Sortformer
4-speaker v2.1, so EMEL should support one exact artifact through native ARM execution and
prove the diarization output contract.

## Table Stakes

| Category | Why It Belongs In v1 | Proof / Testing Implication |
|----------|-----------------------|-----------------------------|
| Pinned Sortformer GGUF fixture | Maintained support needs one exact artifact, URL, checksum, and expected metadata. | Fixture README/provenance plus loader tests for accept/reject behavior. |
| Explicit Sortformer model family | The repo currently has no diarization model family. Aliasing this onto generation or embeddings would be misleading. | Model-family validation tests for architecture metadata, tensor families, and stream parameters. |
| Diarization request actor | The user-facing behavior is who spoke when, not embedding extraction. | SML actor tests for valid request, invalid request, callback/error paths, and unexpected events. |
| Mono 16 kHz PCM contract | Upstream accepts mono 16,000 Hz audio. Keeping the input in memory avoids media decode scope creep. | Negative tests for sample rate, channel count, buffer shape, and capacity. |
| Native Sortformer runtime | ARM support requires EMEL-owned execution, not a tool-only Python/ONNX/NeMo fallback. | Runtime tests that execute through state machines and public event interfaces. |
| `T x 4` probability output | Upstream defines four speaker probabilities per frame. This is the core deterministic output. | Snapshot-friendly probability records with 0.08-second frame semantics. |
| Segment decoding | Operators need bounded speaker intervals, not only raw probabilities. | Tests for monotonic timestamps, stable speaker labels, threshold/overlap behavior, and deterministic JSONL output. |
| Parity proof and benchmark | A new runtime family is not maintained until it has reference evidence and ARM publication. | Lane-isolated reference baseline and one maintained ARM benchmark. |

## Explicit Deferments

| Deferred Surface | Why Defer | What To Do Instead |
|------------------|-----------|--------------------|
| Generic diarization family support | The first slice should prove one maintained model. | Accept only the pinned openresearchtools v2.1 GGUF contract. |
| Live microphone or streaming service API | The runtime can prove streaming profile behavior on deterministic PCM fixtures first. | Use chunk/profile parameters in tests and benchmarks without public live I/O. |
| WAV/MP3 decode, resample, channel mixing | Media ingestion is a separate product surface. | Accept mono `float32` PCM at 16 kHz only. |
| ASR transcription or speaker identity enrollment | Sortformer diarization answers who spoke when, not what they said or who they are. | Emit speaker probability and segment records only. |
| ONNX/CoreML/CUDA/Metal/Vulkan backends | ARM native support should land in EMEL-owned runtime first. | Keep external runtimes only as reference tooling if needed. |
| Broad quant matrix | Quantized operand truth should be handled per maintained artifact. | Document the exact GGUF quantization and required kernels in Phase 81. |

## MVP Recommendation

1. Pin the exact Sortformer GGUF artifact and architecture contract.
2. Add a diarization-owned actor and mono PCM request/result contract.
3. Implement native Sortformer execution for the maintained ARM slice.
4. Emit deterministic probability and segment records.
5. Close with reference proof, ARM benchmark, and operator docs.

## Testing And Proof Notes

- Do not drive parity through embedding or generation compare contracts unless a deliberate
  diarization compare contract is added.
- Keep EMEL/reference lanes isolated. Reference code may use NeMo or another trusted path, but
  EMEL must not share model, cache, tensors, runtime state, or output buffers with it.
- Store or generate only tiny deterministic fixtures suitable for repo gates. Large audio assets
  should remain documented external fixtures unless explicitly approved.
- Use `scripts/quality_gates.sh` after implementation phases, and keep focused doctest files scoped
  to one machine/system/behavior.

## Requirement-Shaped Categories

- Fixture Identity And Provenance
- Sortformer Model Acceptance And Validation
- Diarization Request And Audio Frontend
- Native ARM Sortformer Execution
- Speaker Probability And Segment Output
- Lane-Isolated Reference Proof
- ARM Benchmark Publication
- Operator Support Boundaries

## Sources

- https://huggingface.co/openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf
- https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
- https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/
- `src/emel/model/architecture/detail.cpp`
- `src/emel/embeddings/generator/context.hpp`
- `tests/embeddings/audio_embedding_lane_tests.cpp`
- `tools/bench/**`

---
*Feature research for: ARM Sortformer diarization GGUF support in EMEL*
*Researched: 2026-04-22*
