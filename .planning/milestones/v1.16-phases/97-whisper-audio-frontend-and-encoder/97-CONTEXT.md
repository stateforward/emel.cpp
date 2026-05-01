# Phase 97: Whisper Audio Frontend And Encoder - Context

**Gathered:** 2026-04-26
**Status:** Implemented and verified
**Mode:** Autonomous local continuation after user instruction to kill the agent and do the work
inline

<domain>
## Phase Boundary

Phase 97 must add the maintained Whisper audio request surface, mel feature preparation, and full
encoder execution path for the v1.16 maintained Whisper tiny GGUF family. The phase stops at
deterministic encoder-state evidence; decoder tokens and transcript output remain Phase 98.

ROADMAP success criteria:
1. Public request contract accepts deterministic mono 16 kHz PCM and rejects invalid audio.
2. Mel feature preparation uses loaded Whisper mel filters and bounded EMEL-owned buffers.
3. Encoder convolution, positional embedding, attention, and feed-forward execution run through
   `src/` code.
4. Deterministic encoder-state evidence exists for the pinned audio fixture.

</domain>

<decisions>
## Implementation Decisions

### Locked User Direction

- User rejected ad hoc implementations that avoid kernel work.
- User selected full encoder end-to-end with radix-2 STFT/FFT.
- No skeleton encoder, one-block encoder, direct-DFT shortcut, tool-local compute, reference
  bootstrap, whole-tensor dequantize-to-f32 hot-path fallback, or simplified substitute algorithm
  is acceptable.
- Missing numeric work must land in the owning kernel surface first; non-kernel layers may only
  validate, bind buffers, dispatch, orchestrate phases, and produce evidence.

### Maintained Model Scope

- Phase 95 narrowed EMEL-loadable Whisper tiny GGUF support to `{q8_0, q4_0, q4_1}` for the
  pinned `oxide-lab/whisper-tiny-GGUF` commit.
- Phase 96 added native ARM q4_0/q4_1/q8_0 matmul support and explicit SML routing proof.
- Phase 97 must not regress that scope by building an encoder path that only works for a
  synthetic f32 or q80-only operand contract unless the ROADMAP is explicitly narrowed again by
  the user.

### Runtime Shape

- The request contract should be an EMEL-owned public ASR/Whisper encoder request accepting
  immutable mono 16 kHz PCM, model contract reference, preallocated/bounded output buffers, and an
  immediate same-RTC callback or error code.
- Any SML runtime behavior choice belongs in `guards.hpp` and `sm.hpp`; actions execute already
  selected phases.
- Per-dispatch PCM/model/output pointers must stay in events and typed internal `_done`/`_error`
  events, not persistent context.

</decisions>

<code_context>
## Source-Backed Code Insights

- `src/emel/model/whisper/detail.cpp` validates the tiny execution contract and binds family views
  for `mel_filters`, `model.encoder.*`, and `model.decoder.*`. It validates key shapes such as
  `mel_filters` `{201,80}`, `model.encoder.conv1.weight` `{3,80,384}`, and
  `model.encoder.embed_positions.weight` `{384,1500}`.
- `src/emel/kernel/detail.hpp` currently implements generic scalar `op_mul_mat`, `op_soft_max`,
  selected unary subops (`abs`, `neg`, `relu`, `exp`), and `op_flash_attn_ext`. It lists many GGML
  op events but does not currently execute the Whisper-required `op_norm`, GELU, convolution, or
  mel/STFT preparation path.
- `src/emel/kernel/aarch64/actions.hpp`, `guards.hpp`, `context.hpp`, and `sm.hpp` now contain the
  Phase 96 native q4_0/q4_1/q8_0 vector matmul lanes and dispatch counters.
- `src/emel/diarization/sortformer/encoder/feature_extractor/detail.cpp` contains a radix-2 FFT
  and mel feature extraction implementation, but it is diarization-component-local. It is a useful
  arithmetic reference only; extending it for Whisper would violate the Phase 97 kernel-locality
  direction.
- `src/emel/embeddings/generator/detail.hpp` contains an audio runtime with FFT table setup and
  mel feature preparation, but it is embedding-generator-local. It must not become a Whisper
  shortcut path.
- The only staged real Whisper model fixture at the moment is
  `tests/models/model-tiny-q80.gguf`; q4_0/q4_1 fixture tests are optional when those larger files
  are absent. The only staged mono 16 kHz WAV fixture found locally is
  `tests/fixtures/diarization/ami_en2002b_mix_headset_137.00_152.04_16khz_mono.wav`. Phase 97 must
  either pin a Whisper-owned audio fixture with checksum metadata or explicitly reuse that fixture
  under a source-backed test contract.

</code_context>

<specifics>
## Specific Ideas

- Add failing tests before implementation:
  - invalid audio contracts reject non-16 kHz, non-mono, empty, NaN/Inf, and insufficient buffer
    requests.
  - mel preparation consumes the loaded `mel_filters` tensor, not generated filters.
  - full encoder dispatch records every required kernel phase and produces a deterministic digest
    for the pinned audio fixture and model.
- Kernel-owned radix-2 FFT-core mel, conv1d, layernorm, GELU, positional add, q/k/v/out
  projection, attention, and MLP helpers were implemented before wiring the Whisper actor.
- Keep one-time heap allocation during runtime construction only; dispatch must reuse bounded
  buffers and avoid dynamic allocation.

</specifics>

<deferred>
## Deferred Ideas

- Decoder tokens, transcript output, tokenizer behavior, and timestamp handling are Phase 98.
- Lane-isolated `whisper.cpp` parity records are Phase 99.
- Benchmark publication and ARM performance closure are Phases 100 and 101.

</deferred>

---
*Phase: 97-whisper-audio-frontend-and-encoder*
*Context gathered: 2026-04-26*
