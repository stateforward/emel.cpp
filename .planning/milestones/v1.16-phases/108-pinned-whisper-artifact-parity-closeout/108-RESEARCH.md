---
phase: 108
status: complete
researched: 2026-04-27
requirements:
  - PARITY-01
  - CLOSE-01
decision:
  approved_option: B
  approved_by: user
  approved_at: 2026-04-27
---

# Phase 108 Research

## Finding

The pinned Phase 99 model path is a whisper.cpp legacy artifact with `lmgg` magic despite the
`.gguf` filename:

```sh
build/whisper_reference/whisper-tiny-q8_0-whispercpp.gguf
```

The maintained EMEL GGUF loader expects `GGUF`, so direct GGUF parsing of this artifact failed
before the recognizer runtime could execute. The earlier bridge in
`tools/bench/whisper_normalize_model.py` was useful evidence, but it was bench-owned and could not
be represented as final maintained runtime ingestion.

## Approved Contract

The user approved Option B on 2026-04-27: implement a source-owned conversion path as the final
v1.16 closeout contract.

The maintained contract is:

- the compare/default benchmark path uses the pinned source model path for both lanes;
- if the EMEL lane sees a legacy Whisper `lmgg` image, it normalizes that image through
  `src/emel/model/whisper/detail.cpp`;
- the resulting in-memory GGUF image is consumed by the existing EMEL GGUF loader and maintained
  speech encoder/decoder/tokenizer runtime;
- the bench-only normalizer is not used by the default maintained compare flow.

## Source-Backed Evidence

- `src/emel/model/whisper/detail.hpp` exposes
  `is_legacy_lmgg_whisper(...)` and `normalize_legacy_lmgg_to_gguf(...)`.
- `src/emel/model/whisper/detail.cpp` parses the legacy `lmgg` header, hparams, mel filter,
  vocab, and tensors, maps names/dimensions into the maintained Whisper GGUF tensor contract, and
  emits an in-memory GGUF image.
- `tools/bench/whisper_emel_parity_runner.cpp` reads the pinned model bytes, invokes the
  source-owned conversion when needed, and then drives the existing GGUF loader plus the maintained
  speech encoder/decoder/tokenizer surface.
- `scripts/bench_whisper_compare.sh` defaults `emel_model="$model_path"` and removes the old
  normalized-output directory when no override is provided.
- `tests/model/loader/lifecycle_tests.cpp` covers conversion of a synthetic legacy Whisper
  artifact into a GGUF image accepted by the GGUF loader probe.

## Result

Default compare now records:

- `comparison_status`: `exact_match`
- EMEL transcript: `[C]`
- reference transcript: `[C]`
- model SHA: `9ade048c9d3692b411572a9a8ad615766168e62fb1d4c234973825a377c71984`
- audio SHA: `695ac1b2c85a0419b6ee052ef90cd09cd0c5688a1445aea735b66883d199e803`
- `model_normalization`: `{}`

The empty `model_normalization` object confirms the default compare path no longer uses the
bench-side normalized bridge artifact.
