---
phase: 125
status: pass
verified: 2026-04-28
---

# Verification 125

## Supersession Notice

Passed for its historical recognizer-closeout scope. Phase 126 superseded this verification by
making recognizer route behavior explicit in SML, Phase 127 superseded it by removing decoder
production dependency on encoder detail, and Phase 128 owns evidence-ledger cleanup.

## Source-Backed Audit Evidence

- `tools/bench/whisper_emel_parity_runner.cpp:378-392` creates `speech_recognizer::sm`, binds
  `recognizer_routes::whisper::backend()`, and calls `recognizer.process_event(initialize_ev)`.
- `tools/bench/whisper_emel_parity_runner.cpp:419-442` constructs a public recognizer
  `event::recognize`, supplies caller-owned storage spans, and calls
  `recognizer.process_event(recognize_ev)`.
- A bypass grep over the maintained runner for direct `encoder::whisper`, `decoder::whisper`,
  `speech/encoder/whisper`, `speech/decoder/whisper`, `decode_token_ids`, and old
  `emel.speech.whisper.encoder_decoder` metadata returned no matches.
- `src/emel/speech/recognizer_routes/whisper/detail.cpp` validates pinned tokenizer SHA, tokenizer
  control tokens, ASR decode-policy support, maintained model contract support, and storage
  readiness before route execution.
- Compare and benchmark summaries both publish `backend_id=emel.speech.recognizer.whisper` and
  `runtime_surface=speech/recognizer+speech/recognizer_routes/whisper`.

## Focused Commands

- `scripts/check_domain_boundaries.sh` passed.
- Generic recognizer leak grep returned no matches.
- Forbidden-root grep returned no matches.
- `build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/speech/recognizer/*'`
  passed: 7 test cases / 125 assertions.
- `build/audit-native/emel_tests_bin --no-breaks --test-case='whisper_recognizer*'`
  passed: 1 test case / 356 assertions.
- `build/whisper_compare_tools/whisper_benchmark_tests --no-breaks`
  passed: 10 test cases / 148 assertions.

## Maintained Evidence

- `build/whisper_compare/summary.json`: `exact_match`, `ok`,
  `emel.speech.recognizer.whisper`,
  `speech/recognizer+speech/recognizer_routes/whisper`, transcript `[C]`.
- `build/whisper_benchmark/benchmark_summary.json`: `ok`, `ok`,
  `emel.speech.recognizer.whisper`,
  `speech/recognizer+speech/recognizer_routes/whisper`, transcript `[C]`, EMEL mean
  `59,106,792 ns`, reference mean `59,958,847 ns`.

## Full Closeout Gate

Command:

```bash
EMEL_QUALITY_GATES_SCOPE=full \
EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare:whisper_single_thread \
scripts/quality_gates.sh
```

Result: passed.

- Full coverage test set passed: 12/12 shards.
- Coverage passed: line `90.8%`, branch `55.6%`.
- Paritychecker tests passed.
- Fuzz smoke passed for GGUF parser, GBNF parser, Jinja parser, and Jinja formatter corpora.
- Recognizer-backed compare passed with `status=exact_match reason=ok`.
- Recognizer-backed single-thread benchmark passed with `benchmark_status=ok reason=ok`.
- Docs generation completed.

Existing Zig/locale/cache/OpenMP/linker warnings were printed by reference/build helper lanes and
did not fail the gate.
