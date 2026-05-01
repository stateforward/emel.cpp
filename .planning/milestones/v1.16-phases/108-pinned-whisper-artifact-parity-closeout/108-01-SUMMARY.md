---
phase: 108
plan: 01
status: complete
completed: 2026-04-27
requirements_completed:
  - PARITY-01
  - CLOSE-01
decision:
  approved_option: B
  approved_by: user
  approved_at: 2026-04-27
---

# Phase 108 Summary

## Outcome

Phase 108 completed the reopened v1.16 parity closeout using the user-approved Option B contract:
a source-owned conversion path for the pinned legacy Whisper `lmgg` artifact.

The maintained compare flow now points the EMEL lane at the pinned Phase 99 model path by default.
When that image is legacy `lmgg`, the EMEL runner invokes `src/emel/model/whisper` conversion and
then drives the existing GGUF loader plus the speech encoder/decoder/tokenizer runtime surface.
The old bench-side normalizer is not part of the default maintained compare path.

## Changes

- Added `is_legacy_lmgg_whisper(...)` and `normalize_legacy_lmgg_to_gguf(...)` in
  `src/emel/model/whisper/detail.hpp`.
- Implemented legacy Whisper parsing, tensor-name canonicalization, dimension canonicalization,
  and in-memory GGUF emission in `src/emel/model/whisper/detail.cpp`.
- Updated `tools/bench/whisper_emel_parity_runner.cpp` to convert legacy pinned model bytes before
  binding the model through the GGUF loader and maintained speech runtime surface.
- Updated `scripts/bench_whisper_compare.sh` so the default EMEL model is the pinned source model
  path and stale bench-normalized output is removed.
- Added regression coverage in `tests/model/loader/lifecycle_tests.cpp`.

## Evidence

Default compare:

```sh
scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build
```

Result:

- `comparison_status`: `exact_match`
- EMEL transcript: `[C]`
- reference transcript: `[C]`
- model SHA: `9ade048c9d3692b411572a9a8ad615766168e62fb1d4c234973825a377c71984`
- audio SHA: `695ac1b2c85a0419b6ee052ef90cd09cd0c5688a1445aea735b66883d199e803`
- tokenizer SHA: `dfc530298b6fbed1a97c6472c575b026453706e2a204c7f7038f2c9d208b0759`
- `model_normalization`: `{}`

Full closeout gate:

```sh
EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare \
  scripts/quality_gates.sh
```

Result: passed. Coverage reported 90.6% line and 55.6% branch; paritychecker, fuzz, Whisper
compare, and docsgen all completed.
