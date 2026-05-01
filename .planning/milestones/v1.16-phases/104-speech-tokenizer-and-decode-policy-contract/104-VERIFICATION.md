---
phase: 104
status: passed
verified: 2026-04-27
claims:
  - tokenizer_contract_introduced
  - tokenizer_backed_transcript_publication_evidence
remaining_scope:
  - TOK-01 remains Phase 107
  - TOK-02 remains Phase 107
  - POLICY-01 remains Phase 107
---

# Phase 104 Verification

## Verdict

Phase 104 introduced the speech-domain Whisper tokenizer contract and tokenizer-backed transcript
publication evidence. It does not close the active tokenizer/decode hardening requirements.
TOK-01 remains Phase 107, and final decode policy hardening remains Phase 107 scope.

## Source Evidence

| Claim | Evidence |
|-------|----------|
| Maintained tokenizer asset exists | `tests/models/tokenizer-tiny.json` |
| Tokenizer SHA is the pinned Phase 104 asset | `dfc530298b6fbed1a97c6472c575b026453706e2a204c7f7038f2c9d208b0759` |
| Speech tokenizer contract exists | `src/emel/speech/tokenizer/whisper/detail.hpp` |
| Tokenizer tests prove `[Bell]` detokenization from the asset | `tests/speech/tokenizer/whisper_tests.cpp` |
| Kernel does not hardcode transcript text or numeric placeholders | `! rg -q '"\\[Bell\\]"|"token:"' src/emel/kernel/whisper` |

## Re-run Commands

```sh
test -f tests/models/tokenizer-tiny.json
shasum -a 256 tests/models/tokenizer-tiny.json | \
  rg -Fq 'dfc530298b6fbed1a97c6472c575b026453706e2a204c7f7038f2c9d208b0759'
test -f src/emel/speech/tokenizer/whisper/detail.hpp
! rg -q '"\[Bell\]"|"token:"' src/emel/kernel/whisper
```

## Boundary

Phase 107 scope remains: enforce the tokenizer checksum before maintained dispatch, make prompt
sequence/language/task/timestamp/suppression policy explicit, and remove dispatch-time recognizer
allocation. Phase 104 verification must not be used as `TOK-01`, `TOK-02`, `POLICY-01`,
`PARITY-01`, or `CLOSE-01` closeout evidence.
