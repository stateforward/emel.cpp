---
phase: 127
plan: 01
status: complete
completed: 2026-04-28
requirements-completed:
  - SPEECH-01
  - POLICY-01
  - CLOSE-01
---

# Phase 127 Summary: Whisper Decoder Ownership Gap Closure

## Outcome

Phase 127 closed the milestone audit blocker where the maintained Whisper decoder path executed
through encoder-owned detail code.

## Completed Work

- Rewired `src/emel/speech/decoder/whisper/actions.hpp` to use decoder-owned detail helpers for
  decoder runtime execution.
- Rewired `src/emel/speech/decoder/whisper/guards.hpp` to use decoder-owned tensor helper
  predicates.
- Added a decoder ownership regression in
  `tests/speech/decoder/whisper/lifecycle_tests.cpp` that checks production decoder files do not
  include or alias encoder detail and that decoder detail owns the decode entrypoints.
- Preserved recognizer-backed exact parity and benchmark publication through
  `emel.speech.recognizer.whisper`.

## Evidence

- Decoder dependency grep over `src/emel/speech/decoder/whisper` returned no
  `encoder/whisper/detail` or `encoder::whisper::detail` matches.
- Focused decoder tests passed: 5 test cases, 1431 assertions.
- Focused recognizer test passed: 1 test case, 356 assertions.
- SML behavior-selection scan passed over recognizer, Whisper route, encoder, decoder, and
  tokenizer paths.
- Domain-boundary check passed and forbidden model-family root grep returned no matches.
- Scoped quality gate passed with speech-shard build/test, changed-source coverage, exact
  Whisper compare, and single-thread Whisper benchmark.

## Notes

The 3-iteration benchmark sample was noisy and initially reported a wall-time regression. The
10-iteration gate sample passed with EMEL mean `58,537,483 ns` versus reference mean
`60,435,595 ns`, same model hash, and exact `[C]` transcript parity.
