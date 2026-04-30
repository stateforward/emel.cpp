---
phase: 123
status: passed
verified: 2026-04-28
requirements:
  - SPEECH-01
  - TOK-01
  - TOK-02
  - POLICY-01
---

# Phase 123 Verification

## Verdict

Passed. The public speech recognizer actor now initializes and recognizes through a generic backend
contract, and the maintained Whisper route is available without leaking Whisper identifiers into the
generic recognizer tree.

## Evidence

| Check | Result |
|-------|--------|
| `cmake --build build/audit-native --target emel_tests_bin -j 6` | passed |
| `build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/speech/recognizer/*'` | passed, 7 test cases / 120 assertions |
| `build/audit-native/emel_tests_bin --no-breaks --test-case='whisper_recognizer*'` | passed, 1 test case / 356 assertions |
| `build/audit-native/emel_tests_bin --no-breaks --test-case='speech whisper tokenizer*'` | passed, 4 test cases / 41 assertions |
| `scripts/check_domain_boundaries.sh` | passed |
| `rg -n "emel/whisper\|namespace emel::whisper\|kernel/whisper\|kernel::whisper" src tests CMakeLists.txt` | no matches |
| `rg -n "whisper\|model/whisper\|speech/tokenizer/whisper\|speech/encoder/whisper\|speech/decoder/whisper\|model::whisper\|tokenizer::whisper\|encoder::whisper\|decoder::whisper" src/emel/speech/recognizer tests/speech/recognizer` | no matches |
| Changed-file scoped `scripts/quality_gates.sh` with `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare` | passed |

## Source-Backed Route Truth

`emel::speech::recognizer::sm` no longer has hardcoded unsupported guards for the maintained
route. Initialization binds a generic backend when tokenizer and model support guards pass. A
recognition dispatch then runs explicit recognizer SML phases for encode, decode, and detokenize
before publishing transcript size, token, confidence, frame, width, and digest outputs.

The Whisper-specific work lives under `src/emel/speech/recognizer_routes/whisper/**`, not in the
generic recognizer tree. That route validates the pinned tokenizer SHA
`dfc530298b6fbed1a97c6472c575b026453706e2a204c7f7038f2c9d208b0759`, validates the maintained
Whisper model contract, drives speech-owned encoder/decoder public events with `process_event(...)`,
and publishes transcript text through `speech/tokenizer/whisper::decode_token_ids`.

## Remaining Gap

The maintained compare and benchmark tools still use the old direct runner path. Phase 124 owns
cutting those proof entrypoints over to this public recognizer route and updating their
recognizer-backed runtime metadata.
