# Pitfalls Research

**Domain:** ARM support for `openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf`
**Researched:** 2026-04-22
**Confidence:** MEDIUM

## Critical Pitfalls

### 1. Treating Diarization As Embedding Or Generation

**What goes wrong:** Sortformer is routed through existing embedding or generation surfaces because
they already handle audio or GGUF files.

**Why it is risky:** the maintained behavior is diarization: `T x 4` speaker activity and segment
records. Reusing the wrong domain would hide the output contract.

**How to prevent it:** create a diarization-owned actor and keep embedding/generation tooling as
references only where appropriate.

### 2. Letting Reference Runtime Become EMEL Runtime

**What goes wrong:** the implementation shells out to Python, NeMo, ONNX, llama.cpp, or ggml from
the EMEL lane and calls the milestone supported.

**Why it is risky:** the user asked for ARM support in this codebase. Tool-only compute fallback is
not native support.

**How to prevent it:** keep external runtimes confined to reference lanes and require `src/`
runtime execution for EMEL.

### 3. Hiding Streaming/Profile/Cache Routing In Detail Helpers

**What goes wrong:** chunk profiles, cache readiness, or error outcomes are selected by `if`/switch
logic in actions/detail helpers.

**Why it is risky:** AGENTS rules require runtime behavior choice in guards/states/transitions.

**How to prevent it:** model model-readiness, profile selection, cache readiness, and validation
outcomes in `sm.hpp` with guard predicates.

### 4. Overloading The Milestone With Media Ingestion

**What goes wrong:** WAV/MP3 decode, resampling, channel mixing, live microphone capture, or socket
streaming gets added before the runtime is proven.

**Why it is risky:** those are useful features, but they widen the milestone far beyond one
maintained runtime slice.

**How to prevent it:** keep the v1 input contract to mono `float32` PCM at 16,000 Hz.

### 5. Publishing Segments Without Probability Truth

**What goes wrong:** implementation only publishes speaker segments and discards or ignores the
frame probability matrix.

**Why it is risky:** upstream defines tensor outputs as `T x 4` probabilities with 0.08-second
frame semantics. Segment decoding is downstream of that model output.

**How to prevent it:** require both probability matrix proof and segment-record proof.

### 6. Calling Parity Without Lane Isolation

**What goes wrong:** EMEL/reference comparison shares model state, tokenizer/audio frontend state,
cache buffers, tensor memory, or output objects.

**Why it is risky:** prior milestone rules explicitly prohibit shared runtime state between EMEL
and reference lanes.

**How to prevent it:** use separate setup and result files for EMEL and reference lanes, and make
the compare step consume serialized outputs only.

### 7. Underestimating Dynamic Streaming Operations

**What goes wrong:** the plan assumes Sortformer can be treated as a simple static graph.

**Why it is risky:** public ONNX export discussion highlights dynamic slicing and runtime-dependent
indices in streaming Sortformer modules. EMEL needs explicit orchestration for those behaviors.

**How to prevent it:** surface streaming/cache operations as first-class runtime contracts and SML
states instead of assuming a one-shot static inference path.

## Pitfall To Phase Mapping

| Pitfall | Phase To Address |
|---------|------------------|
| Wrong domain surface | Phase 82 |
| Reference runtime leakage | Phase 83 and Phase 85 |
| Hidden runtime routing | Phase 83 |
| Media ingestion scope creep | Phase 82 |
| Segment-only proof | Phase 84 |
| Lane isolation drift | Phase 85 |
| Static-graph assumption | Phase 81 and Phase 83 |

## Warning Signs

- Docs say "speech support" without naming the exact Sortformer GGUF artifact.
- A diarization path returns embeddings, transcripts, or arbitrary labels instead of `T x 4`
  probabilities and four-speaker segments.
- A benchmark wrapper produces the only working output path.
- Tests pass because Python/NeMo generated the EMEL result.
- Runtime code stores request/output pointers or phase counters in context.
- Sortformer cache/profile handling appears in `detail.cpp` as routing logic.

## Sources

- https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
- https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/
- https://github.com/NVIDIA-NeMo/NeMo/issues/15077
- `AGENTS.md`
- `src/emel/embeddings/generator/detail.hpp`
- `tools/bench/**`

---
*Pitfalls research for: ARM Sortformer diarization GGUF support in EMEL*
*Researched: 2026-04-22*
