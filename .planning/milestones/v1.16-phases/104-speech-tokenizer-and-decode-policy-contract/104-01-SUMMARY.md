---
phase: 104
plan: 1
status: complete
completed: 2026-04-26
---

# Plan 104.1 Summary

## Completed

- Pinned `tests/models/tokenizer-tiny.json` with checksum
  `dfc530298b6fbed1a97c6472c575b026453706e2a204c7f7038f2c9d208b0759`.
- Updated `scripts/setup_whisper_cpp_reference.sh` to fetch and verify the tokenizer asset.
- Added `src/emel/speech/tokenizer/whisper/detail.hpp` as the speech-domain tokenizer contract
  for Whisper control-token roles and minimal tokenizer-backed detokenization.
- Updated the EMEL Whisper compare and single-thread benchmark runners to require the tokenizer
  asset and avoid `token:<id>` placeholder transcript publication.
- Added `tests/speech/tokenizer/whisper_tests.cpp` proving control roles and `[Bell]`
  detokenization are backed by the pinned tokenizer JSON.
- Added `speech` quality-gate shard mapping for `src/emel/speech/**`.

## Verification

- `build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/speech/*'`
- `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build`
- Scoped `scripts/quality_gates.sh` with `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_single_thread`
  passed after this implementation.

## Remaining Blocker

Exact transcript parity remains open for Phase 105. Current EMEL output is tokenizer-backed but
still differs from the pinned `whisper.cpp` reference: EMEL selects EOT/empty transcript and the
reference transcript is `[Bell]`.
