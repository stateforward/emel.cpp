---
phase: 126
plan: 01
status: complete
completed: 2026-04-28
requirements-completed:
  - CLOSE-01
---

# Summary 126.1

## Completed

- Removed the generic recognizer `runtime_backend` function-pointer contract and the initialize
  backend pointer.
- Replaced runtime backend dispatch with a compile-time recognizer route policy that is visible in
  `emel::speech::recognizer::sm` guards/actions.
- Added the Whisper route policy under `src/emel/speech/recognizer_routes/whisper/**`, keeping
  Whisper/model-family names out of the generic recognizer public boundary.
- Updated recognizer tests, the Whisper fixture integration test, and the maintained Whisper
  parity runner to instantiate the explicit route policy.

## Verification Highlights

- SML behavior-selection scan passed on the recognizer, Whisper route, encoder, decoder, and
  tokenizer paths.
- Domain-boundary script passed; forbidden-root and generic recognizer leak greps returned no
  matches.
- Focused recognizer doctest passed: 7 test cases / 121 assertions.
- Whisper recognizer fixture doctest passed: 1 test case / 356 assertions.
- CTest speech and Whisper shards passed.
- Changed-file scoped quality gate passed with line coverage `96.9%`, branch coverage `55.7%`,
  Whisper compare exact-match, Whisper single-thread benchmark, and docs generation.
- Full closeout quality gate passed with 12/12 test shards, line coverage `90.8%`, branch
  coverage `55.6%`, paritychecker, fuzz smoke, Whisper compare, Whisper single-thread
  benchmark, and docs generation.

## Residual Work

- `ctest --test-dir build/audit-native -R '^lint_snapshot$' --output-on-failure` still fails
  because the existing lint snapshot baseline is stale for unrelated Whisper decoder/encoder files
  and would need an explicit snapshot update approval. The Phase 126 changed files were formatted.
