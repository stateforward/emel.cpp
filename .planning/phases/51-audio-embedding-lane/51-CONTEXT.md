---
phase: 51
slug: audio-embedding-lane
created: 2026-04-14
status: ready
---

# Phase 51 Context

## Phase Boundary

Phase 51 adds the maintained in-memory TE audio lane on top of the now-complete text and vision
lanes. The scope is intentionally narrow: accept one documented mono PCM float32 memory contract,
run TE audio preprocessing plus the TE audio encoder/projection path, and return a normalized
shared-space embedding. This phase does not widen into WAV/MP3 decode, generic audio import, or
public API expansion.

## Implementation Decisions

### Runtime Surface
- Extend the existing `src/emel/embeddings/generator/` actor with an audio request surface for the
  maintained slice rather than introducing public API churn in the same phase.
- Keep request modality routing explicit in the state machine; do not hide text/image/audio choice
  in helper branching.
- Preserve the locked architectural direction (`audio/encoders/...`) as future cleanup, but keep
  this phase’s maintained runtime cut focused on truthful end-to-end behavior first.

### Audio Truth
- Drive the audio lane from the pinned `TE-75M-q8_0.gguf` contract only.
- Treat `omniembed.audio_encoder_name=efficientat_mn20_as` as the authoritative encoder identity.
- Derive the encoder-native frontend from the upstream EfficientAT source:
  preemphasis, STFT, Kaldi mel-bank projection, log compression, and fast normalization.
- Keep the maintained outer payload contract pinned to the proof corpus:
  mono PCM `float32` at `16000` Hz. Internal resampling to the encoder-native `32000` Hz frontend
  is allowed as bounded inference-time preprocessing for this phase.
- Consume the pooled `1920`-channel EfficientAT feature vector and project it through
  `audio_projection.*`; do not route through the AudioSet classifier logits head.

### Scope Guardrails
- Accept only the documented in-memory audio payload contract from
  `tests/embeddings/fixtures/te75m/README.md`.
- Keep audio preprocessing deterministic and bounded; generic file decode/resample/transcode
  productization stays out of scope.
- Limit proof in this phase to maintained flow behavior, normalization, and invalid-payload
  rejection. Golden cross-modal comparison remains Phase 53 work.

## Existing Code Insights

### Reusable Assets
- `src/emel/model/omniembed/detail.cpp` already validates and exposes `audio_encoder.*` and
  `audio_projection.*` families through the execution contract.
- `src/emel/model/data.hpp` already carries the audio metadata Phase 51 needs to extend:
  `encoder_name`, `embedding_length`, and `projection_dim`.
- `src/emel/embeddings/generator/` now provides the shared embedding-session actor, error/callback
  plumbing, normalized publication contract, and the recently added image-lane extension pattern.
- `tests/embeddings/fixtures/te75m/README.md` already defines the canonical maintained audio
  anchor as a `0.25`-second `440 Hz` mono sine wave synthesized directly in memory.

### Derived Audio Runtime Shape
- The pinned GGUF carries `audio_encoder.features.0..16.*`, `audio_encoder.classifier.*`, and
  `audio_projection.*`.
- The feature stack matches EfficientAT `mn20_as` / MobileNetV3 width `2.0`:
  stem -> 15 inverted-residual blocks with optional SE -> final `1x1` conv -> global average
  pool.
- Several SE layers are quantized (`Q8_0`) even though the surrounding conv weights are `F16`, so
  the maintained runtime must support mixed dense/quantized SE MLP weights on the native path.

### Hard Constraints
- AGENTS still requires explicit SML routing, bounded actions, and no dispatch-time allocation.
- Phase 51 must not imply broad audio support; it proves one truthful maintained memory contract.
- Quality gates must stay green even if failures surface outside the current lane.

## Specific Ideas

- Add an internal `embed_audio` request to `embeddings/generator` with explicit success/error
  publication and invalid-request rejection.
- Add a narrow maintained audio preprocessor that:
  - validates mono `float32` PCM payload shape
  - resamples the maintained `16 kHz` contract to the encoder-native `32 kHz` rate
  - applies EfficientAT preemphasis, STFT, Kaldi mel-bank projection, log compression, and fast
    normalization into the model-required spectrogram layout
- Implement the EfficientAT `mn20_as` feature tower and audio projection path against the real
  `audio_encoder.*` and `audio_projection.*` tensors, then reuse the existing shared embedding
  publication logic.
- Add maintained tests that prove:
  - normalized `1280`-dimensional audio embedding output
  - explicit invalid audio payload rejection
  - compatibility with the canonical `pure-tone-440hz` anchor

## Deferred Ideas

- WAV/MP3 decoding
- arbitrary external sample-rate and channel-layout product support beyond the maintained contract
- public audio embedding API
- golden-vector parity publication
- repo-wide `audio/encoders` extraction cleanup

## Primary Sources

- `tests/models/TE-75M-q8_0.gguf`
- `tests/embeddings/fixtures/te75m/README.md`
- `src/emel/model/omniembed/detail.cpp`
- `src/emel/model/data.hpp`
- `https://huggingface.co/augmem/TE-75M`
- `https://github.com/fschmid56/EfficientAT`

---
*Phase: 51-audio-embedding-lane*
*Context gathered: 2026-04-14*
