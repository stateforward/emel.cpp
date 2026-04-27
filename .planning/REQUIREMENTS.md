# Requirements: v1.16 Reopened Whisper E2E Closure

## Active Requirements

- [ ] **REOPEN-01** — Reopen v1.16 because `bounded_drift` transcript mismatch is not acceptable
  for the E2E milestone.
- [ ] **SPEECH-01** — Remove the top-level `src/emel/whisper/**` runtime domain; Whisper runtime
  actors must live under the speech recognizer domain while model and kernel ownership stay in
  `model/whisper` and `kernel/whisper`.
- [ ] **TOK-01** — Pin and validate the maintained `tokenizer-tiny.json` asset before dispatch.
- [x] **TOK-02** — Use speech tokenizer/detokenizer machinery for transcript publication; do not
  hardcode fixture transcript text or token-piece mappings in Whisper kernels.
- [ ] **POLICY-01** — Model Whisper ASR decode policy explicitly: prompt sequence, language/task
  roles, timestamp mode, and suppression behavior.
- [x] **PARITY-01** — The maintained EMEL lane must exact-match the pinned `whisper.cpp`
  transcript for the Phase 99 audio/model pair.
- [ ] **CLOSE-01** — Re-run source-backed audit and full relevant quality gates before closing
  the reopened milestone.

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| REOPEN-01 | 109 | Pending |
| SPEECH-01 | 109 | Pending |
| TOK-01 | 111 | Pending |
| TOK-02 | 107 | Complete |
| POLICY-01 | 111 | Pending |
| PARITY-01 | 108 | Complete |
| CLOSE-01 | 112 | Pending |
