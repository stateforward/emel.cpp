---
phase: 129
plan: 01
status: complete
completed: 2026-04-28
requirements-completed: []
---

# Phase 129 Summary: Whisper Detail Helper Deduplication Cleanup

## Outcome

Phase 129 closed the remaining v1.16 audit tech debt by removing duplicate decoder and timestamp
runtime helpers from encoder detail. The maintained decoder runtime remains decoder-owned.

## Completed Work

- Removed decoder-only constants, decoder workspace sizing, decode policy runtime state, decoder
  cross-cache/logit helpers, timestamp-aware token selection, and `run_decoder_sequence` from
  `src/emel/speech/encoder/whisper/detail.hpp`.
- Kept `src/emel/speech/decoder/whisper/detail.hpp` as the sole owner for decoder runtime helper
  execution.
- Moved timestamp helper coverage out of the encoder detail tests and into the existing decoder
  lifecycle test file.
- Added a source regression proving encoder detail does not contain decoder helper names.

## Evidence

- Encoder detail grep for decoder helper names returned no matches.
- Decoder production grep for `encoder/whisper/detail` and `encoder::whisper::detail` returned no
  matches.
- Focused encoder Whisper tests passed: 15 test cases, 2166 assertions.
- Focused decoder Whisper tests passed: 7 test cases, 1436 assertions.
- `ctest` passed for `emel_tests_speech` and `emel_tests_whisper`.
- SML behavior-selection and domain-boundary checks passed.
- Maintained compare passed with `comparison_status=exact_match`, `reason=ok`, and `[C]` on both
  lanes.
- Maintained benchmark passed with 20 iterations, tolerance `20000`, EMEL mean `59,887,720 ns`,
  and reference mean `60,993,289 ns`.
- Changed-file scoped quality gate passed with encoder-detail coverage line `100.0%`, branch
  `50.0%`.

## Snapshot Follow-Up

After explicit user approval, `scripts/lint_snapshot.sh --update` refreshed
`snapshots/lint/clang_format.txt`, and
`ctest --test-dir build/audit-native -R '^lint_snapshot$' --output-on-failure` passed.
