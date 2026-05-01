---
phase: 118
plan: 01
status: complete
completed: 2026-04-27
requirements-completed:
  - SPEECH-01
  - TOK-02
  - POLICY-01
  - PARITY-01
  - PERF-03
---

# Summary 118.1

## Completed

- Added public `any.hpp` surfaces for:
  - `emel::model::whisper`
  - `emel::speech::encoder::whisper`
  - `emel::speech::decoder::whisper`
  - `emel::speech::tokenizer::whisper`
- Updated `tools/bench/whisper_emel_parity_runner.cpp` to use those public surfaces and the
  encoder/decoder actor `process_event(...)` APIs.
- Added a bench doctest that fails if the runner directly includes Whisper model/speech
  `detail.hpp` headers again.
- Narrowed the maintained decode policy from the previously claimed no-timestamps mode to the
  source-backed `timestamp_tokens` mode used by the exact `[C]` compare lane.
- Published decode-policy fields in the EMEL compare JSON:
  `language=english`, `task=transcribe`, `timestamp_mode=timestamp_tokens`,
  `suppress_translate=true`, and `prompt_token_count=3`.
- Updated the lint snapshot after the approved snapshot refresh.

## Verification Highlights

- Compare: `status=exact_match reason=ok`, EMEL transcript `[C]`.
- Benchmark: `status=ok reason=ok`, EMEL mean `62020084 ns`, reference mean `66998708 ns`.
- Focused bench doctests: 10 test cases, 139 assertions passed.
- Speech/Whisper CTest shard: 2/2 tests passed.
- Scoped quality gate passed with line coverage `99.1%` and branch coverage `71.6%`.

## Residual Work

Phase 119 owns final source-backed closeout rerun and the truthful Phase 113 Nyquist-ledger
resolution before v1.16 archival.
