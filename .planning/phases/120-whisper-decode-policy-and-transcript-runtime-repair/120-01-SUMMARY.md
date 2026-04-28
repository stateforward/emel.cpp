---
phase: 120
plan: 01
status: complete
completed: 2026-04-27
requirements-completed:
  - TOK-02
  - POLICY-01
---

# Summary 120.1

## Completed

- Added the full speech-owned Whisper decode control-token policy surface, including
  `no_speech`, `timestamp_begin`, and `space`, and extended policy support checks against those
  fields.
- Changed the decoder public event to carry
  `speech::tokenizer::whisper::asr_decode_policy` by reference instead of prompt-token-only input.
- Removed decoder transcript buffers and `transcript_size` from the decoder event/done surface.
- Added SML policy and generated-token-capacity validation states before selected decoder
  execution.
- Changed greedy timestamp-aware decode to consume policy token fields for EOT, timestamp,
  control-token suppression, and initial-token suppression.
- Removed the hardcoded decoder `token:<id>` transcript writer.
- Updated the maintained Whisper parity runner so text publication remains only through
  `speech/tokenizer/whisper::decode_token_ids`.

## Verification Highlights

- Focused decoder lifecycle tests: 4 test cases, 1419 assertions passed.
- Focused encoder detail tests: 9 test cases, 56 assertions passed.
- Focused tokenizer tests: 4 test cases, 35 assertions passed.
- Bench doctests: 10 test cases, 139 assertions passed.
- Domain-boundary script passed and forbidden-root grep returned no matches.
- Maintained compare: `status=exact_match reason=ok`.
- Maintained single-thread benchmark: `benchmark_status=ok reason=ok`.
- Changed-file scoped quality gate passed with line coverage `99.2%` and branch coverage `50.0%`.

## Residual Work

Phase 121 still needs preserved-baseline Nyquist validation backfill for Phases 94-102. Phase 122
owns final source-backed audit rerun and `CLOSE-01`.
