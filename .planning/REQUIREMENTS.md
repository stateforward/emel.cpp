# Requirements: v1.16 Reopened Whisper E2E And Performance Closure

## Active Requirements

- [x] **REOPEN-01** — Reopen v1.16 because `bounded_drift` transcript mismatch is not acceptable
  for the E2E milestone.
- [x] **SPEECH-01** — Remove the top-level `src/emel/whisper/**` runtime domain; Whisper runtime
  actors live under speech encoder/decoder/tokenizer ownership while model binding stays in
  `model/whisper`.
- [x] **TOK-01** — Pin and validate the maintained `tokenizer-tiny.json` asset before dispatch.
- [x] **TOK-02** — Use speech tokenizer/detokenizer machinery for transcript publication; do not
  hardcode fixture transcript text or token-piece mappings in Whisper kernels.
- [x] **POLICY-01** — Model Whisper ASR decode policy explicitly: prompt sequence, language/task
  roles, timestamp mode, and suppression behavior.
- [x] **PARITY-01** — The maintained EMEL lane must exact-match the pinned `whisper.cpp`
  transcript for the Phase 99 audio/model pair.
- [x] **CLOSE-01** — Re-run source-backed audit and full relevant quality gates before closing
  the reopened milestone, including benchmark evidence that enforces the performance contract.
- [x] **PERF-03** — Restore a source-backed benchmark record where EMEL beats the matched
  single-thread CPU `whisper.cpp` ARM reference lane for the maintained Phase 99 model/audio pair.

**Coverage after latest source-backed audit:** 8 complete, 0 pending.

Phase 128 and Phase 129 are tech-debt cleanup phases created from the milestone audit. They do not
reset active requirement status because the audit found no unsatisfied requirement, integration, or
flow gaps. Phases 128 and 129 are complete.

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| REOPEN-01 | 124 | Complete |
| SPEECH-01 | 127 | Complete |
| TOK-01 | 123 | Complete |
| TOK-02 | 123 | Complete |
| POLICY-01 | 127 | Complete |
| PARITY-01 | 124 | Complete |
| CLOSE-01 | 127 | Complete |
| PERF-03 | 124 | Complete |
