---
phase: 123
plan: 01
status: complete
completed: 2026-04-28
requirements-completed:
  - SPEECH-01
  - TOK-01
  - TOK-02
  - POLICY-01
---

# Summary 123.1

## Completed

- Added a generic, model-family-free recognizer backend contract and caller-owned runtime storage
  spans to the public recognizer event surface.
- Changed the public recognizer SML graph so initialization can bind a supported backend and
  recognition explicitly runs encode, decode, and detokenize phases before publishing outputs.
- Added a variant-named Whisper recognizer route outside `src/emel/speech/recognizer/**`; the
  route validates the pinned tokenizer SHA, validates the maintained Whisper model contract, builds
  speech-owned encoder/decoder events, and publishes transcript text through the speech tokenizer.
- Kept generic recognizer files and generic recognizer tests free of Whisper identifiers, preserving
  `scripts/check_domain_boundaries.sh` as a real guard.
- Added focused tests proving the generic recognizer route works and the Phase 99 model/tokenizer
  path can run through `emel::speech::recognizer::sm`.

## Verification Highlights

- Focused generic recognizer tests: 7 test cases / 120 assertions passed.
- Focused Whisper recognizer fixture test: 1 test case / 356 assertions passed.
- Speech tokenizer tests: 4 test cases / 41 assertions passed.
- `scripts/check_domain_boundaries.sh` passed.
- Forbidden-root grep returned no matches.
- Changed-file scoped quality gate passed with line coverage `100.0%` and branch coverage `64.5%`.
- The Whisper compare suite still exact-matches, but remains bypass-backed until Phase 124 cuts the
  maintained runner over to the recognizer.

## Residual Work

Phase 124 must update `tools/bench/whisper_emel_parity_runner.cpp`,
`tools/bench/whisper_compare.py`, and `tools/bench/whisper_benchmark.py` so parity and benchmark
evidence consume the public recognizer actor instead of the old direct runner lane.
