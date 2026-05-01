---
phase: 107
plan: 01
status: complete
completed: 2026-04-27
requirements-completed:
  - TOK-01
  - TOK-02
  - POLICY-01
---

# Summary 107.1

## Completed

- Added pinned tokenizer asset identity to recognizer initialization.
- Added guard-level rejection for tokenizer SHA mismatches.
- Added explicit Whisper ASR decode policy fields for prompt sequence, language role, task role,
  timestamp mode, and translate suppression.
- Moved recognizer token-sequence workspaces and child actor allocation into context construction,
  before SML dispatch.
- Added tokenizer checksum enforcement to `bench_whisper_compare.sh`, `whisper_compare.py`, and
  `whisper_benchmark.py`.
- Added focused tests for tokenizer SHA rejection and explicit ASR policy prompt tokens.

## Verification

- `cmake --build build/audit-native --target emel_tests_bin --parallel`
- `build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/speech/*,*tests/whisper/*'`
- Changed-file scoped `scripts/quality_gates.sh` with
  `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare`.

## Remaining Work

Phase 108 owns direct pinned-artifact exact transcript parity and final milestone closeout.
