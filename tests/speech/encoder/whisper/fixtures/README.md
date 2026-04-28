# Whisper Encoder Fixture Contract

Phase 97 uses deterministic in-test mono 16 kHz PCM for encoder-state evidence while the
maintained model fixture is `tests/models/model-tiny-q80.gguf`.

The PCM fixture is generated in `tests/speech/encoder/whisper/lifecycle_tests.cpp` as a 320-sample
single-channel 16 kHz waveform. This keeps the public encoder request deterministic and small
enough for full four-layer encoder execution in unit tests. It is not transcript or parity
evidence; `whisper.cpp` parity remains Phase 99.
