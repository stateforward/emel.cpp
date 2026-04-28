---
phase: 122
created: 2026-04-27
source: gsd-plan-milestone-gaps
status: complete
requirements:
  - CLOSE-01
depends_on:
  - 120
  - 121
---

# Phase 122 Context: Whisper Final Gap Closeout Rerun

## Why This Phase Exists

The 2026-04-27 source-backed v1.16 milestone audit found that the maintained Whisper compare and
benchmark lanes still passed, but `CLOSE-01` could not be marked complete while:

- `POLICY-01` was contradicted by prompt-token-only decoder event wiring.
- `TOK-02` was contradicted by the decoder's hardcoded `token:<id>` transcript surface.
- Preserved baseline Phases 94-102 had no Nyquist validation artifacts.

Phase 120 repaired the decode-policy and transcript-runtime source gaps. Phase 121 backfilled
truthful archived-baseline validation artifacts for Phases 94-102. Phase 122 exists only to rerun
the source-backed closeout after those repairs and update the milestone ledgers if the maintained
paths pass.

## Maintained Closeout Surface

- EMEL runtime surface:
  `speech/encoder/whisper+speech/decoder/whisper+speech/tokenizer/whisper`
- Reference lane: pinned `whisper.cpp` v1.7.6 side only.
- Model/audio pair: `whisper/tiny/q8_0/phase99_440hz_16khz_mono`.
- Transcript contract: exact `[C]` parity.
- Benchmark contract: single-thread CPU, warmed multi-iteration wrapper, EMEL mean process wall
  time below matched `whisper.cpp` reference mean.

## Guardrails

- Do not credit artifact-only agreement if source wiring contradicts it.
- Do not use the volatile zero-warmup one-iteration benchmark citation for closeout.
- Do not update snapshot baselines as part of this closeout.
- Keep domain-boundary evidence source-backed with `scripts/check_domain_boundaries.sh` and the
  forbidden-root grep.
