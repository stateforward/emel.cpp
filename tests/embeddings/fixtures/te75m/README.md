# TE-75M Proof Corpus

This directory defines the narrow proof corpus for the maintained `TE-75M-q8_0.gguf` slice.
The corpus is intentionally expressed as deterministic in-memory payload contracts, not PNG/JPEG or
WAV file-decoding requirements.

## Anchor: `red-square`

- Purpose: canonical text-image smoke anchor
- Text prompt file: `red-square.txt`
- Text prompt string: `a red square`
- Image payload contract:
  - format: RGBA `uint8`
  - width: `32`
  - height: `32`
  - layout: row-major
  - every pixel: `[255, 0, 0, 255]`

## Anchor: `pure-tone-440hz`

- Purpose: canonical text-audio smoke anchor
- Text prompt file: `pure-tone-440hz.txt`
- Text prompt string: `a pure 440 hertz sine tone`
- Audio payload contract:
  - format: mono PCM `float32`
  - sample rate: `16000`
  - duration: `0.25` seconds
  - sample count: `4000`
  - amplitude: `0.2`
  - generation rule: `sample[n] = 0.2f * sinf((2.0f * pi * 440.0f * n) / 16000.0f)` for
    `0 <= n < 4000`

## Scope Notes

- These anchors are the maintained truth set for later golden-vector and smoke checks.
- Later phases should synthesize these payloads directly in memory from this manifest.
- Generic image/audio file decoding remains out of scope for the first maintained TE slice.

## Stored Upstream Goldens

- `red-square.text.1280.txt`
- `pure-tone-440hz.text.1280.txt`
- `red-square.image.1280.txt`
- `pure-tone-440hz.audio.1280.txt`

These files are generated from the upstream `augmem/TE-75M` safetensors release on CPU using the
official source encoder families listed on the model card:

- `MongoDB/mdbr-leaf-ir` via `sentence-transformers`
- `mobilenetv4_conv_medium.e180_r384_in12k` via `timm`
- `EfficientAT mn20_as` via the upstream EfficientAT Python implementation

The reproducible generator lives at `generate_upstream_goldens.py` in this same directory.
