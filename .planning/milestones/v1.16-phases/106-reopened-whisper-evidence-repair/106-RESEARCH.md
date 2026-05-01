---
phase: 106
name: Reopened Whisper Evidence Repair
status: research_complete
researched: 2026-04-26
confidence: HIGH
---

# Phase 106 Research

## Goal

Close the artifact and planning-state gaps identified by the v1.16 milestone audit for reopened
Phases 103-105. No source code changes. Output: VERIFICATION and VALIDATION artifacts for Phases
103 and 104, Phase 106 SUMMARY, and updated traceability in REQUIREMENTS.md, ROADMAP.md, and
STATE.md.

## Verified Source Facts

### Phase 103: Speech Recognizer Domain Cleanup

| Claim | Verified | Command |
|-------|----------|---------|
| `src/emel/whisper` domain absent | YES | `test ! -d src/emel/whisper` |
| No `emel/whisper` include paths | YES | `rg "emel/whisper\|emel::whisper" src tests tools CMakeLists.txt` → 0 matches |
| Actors under speech recognizer domain | YES | files live at `src/emel/speech/recognizer/detail/whisper/{encoder,decoder}/**` |
| Namespace is `emel::speech::recognizer::detail::whisper::*` | YES | verified in encoder/sm.hpp |
| SUMMARY.md exists | YES | `.planning/phases/103-.../103-01-SUMMARY.md` |
| VERIFICATION.md missing | YES | audit gap |
| VALIDATION.md missing | YES | audit gap |

**Path discrepancy:** Phase 103 SUMMARY claims actors moved to
`src/emel/speech/recognizer/whisper/encoder/**` with namespace
`emel::speech::recognizer::whisper::*`. Actual path is
`src/emel/speech/recognizer/detail/whisper/encoder/**` with namespace
`emel::speech::recognizer::detail::whisper::encoder`. The SPEECH-01 requirement is still
satisfied — actors ARE under speech recognizer domain and `src/emel/whisper` is gone — but
VERIFICATION.md must record the truthful path, not the summary claim.

### Phase 104: Speech Tokenizer And Decode Policy Contract

| Claim | Verified | Command |
|-------|----------|---------|
| `tests/models/tokenizer-tiny.json` exists | YES | `test -f tests/models/tokenizer-tiny.json` |
| Tokenizer SHA pinned | YES | `dfc530298b6fbed1a97c6472c575b026453706e2a204c7f7038f2c9d208b0759` |
| `src/emel/speech/tokenizer/whisper/detail.hpp` exists | YES | `test -f src/emel/speech/tokenizer/whisper/detail.hpp` |
| No hardcoded `[Bell]` or `token:` in kernel/whisper | YES | `rg '"Bell"\|"token:"' src/emel/kernel/whisper/` → 0 matches |
| SUMMARY.md exists | YES | `.planning/phases/104-.../104-01-SUMMARY.md` |
| VERIFICATION.md missing | YES | audit gap |
| VALIDATION.md missing | YES | audit gap |

**Remaining gap (Phase 107 scope):** `tools/bench/whisper_compare.py` only checks tokenizer file
existence; the pinned SHA `dfc530...` is not enforced at compare/runtime dispatch time. This is
an audit-identified blocker (TOK-01) assigned to Phase 107. Phase 104 VERIFICATION must record
this truthfully: tokenizer asset is introduced and tested but SHA enforcement before dispatch
remains Phase 107 scope.

### Phase 105: Whisper Exact Transcript Parity Closure

All artifacts now exist and are complete:
- `105-01-SUMMARY.md`: `requirements-completed: []`, `superseded-by: [107, 108]`
- `105-VERIFICATION.md`: bridge vs. direct pinned evidence recorded, `direct_pinned_artifact_parity: unsatisfied`
- `105-VALIDATION.md`: `nyquist_compliant: true`, `status: complete`

No Phase 106 work required for Phase 105.

### STATE.md and ROADMAP.md

STATE.md is already truthful:
- `status: reopened`, `phase: 106`, `plan: not started`
- correctly qualifies bridge evidence as not direct pinned-artifact parity
- points to Phase 108 for PARITY-01/CLOSE-01

ROADMAP.md is already truthful:
- Phase 106 is `[ ]` (unchecked, correct)
- Phase 105 marked as transition/evidence work only
- Phases 107/108 own remaining requirements

Both files need updating AFTER Phase 106 execution (mark Phase 106 complete, advance to Phase 107).

## Requirements

| ID | Assigned To | Status | Evidence Path |
|----|-------------|--------|---------------|
| REOPEN-01 | Phase 106 | Pending | Phase 103 VERIFICATION.md (to create) |
| SPEECH-01 | Phase 106 | Pending | Phase 103 VERIFICATION.md (to create) |

## Test Infrastructure

| Shard | Filter | Covers |
|-------|--------|--------|
| speech | `*tests/speech/*,*tests/whisper/*` | recognizer lifecycle, tokenizer whisper tests |
| whisper | `*tests/whisper/*` | encoder, decoder, kernel |

Build: `cmake --build build/audit-native --target emel_tests_bin --parallel`
Run: `build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/speech/*,*tests/whisper/*'`

## What Phase 106 Must Produce

1. `103-VERIFICATION.md` — source-backed evidence with truthful actor path
2. `103-VALIDATION.md` — Nyquist compliance
3. `104-VERIFICATION.md` — tokenizer asset evidence; note SHA enforcement gap (Phase 107)
4. `104-VALIDATION.md` — Nyquist compliance
5. `106-01-SUMMARY.md` — closes REOPEN-01 and SPEECH-01
6. Update `REQUIREMENTS.md` — REOPEN-01, SPEECH-01 to Complete/Phase 103+106
7. Update `ROADMAP.md` — mark Phase 106 `[x]`
8. Update `STATE.md` — advance to Phase 107

## Blockers

None for Phase 106 itself. Downstream blockers:
- Phase 107: tokenizer SHA dispatch enforcement, decode policy contract, dispatch allocation fix
- Phase 108: direct pinned-artifact parity (PARITY-01, CLOSE-01)
