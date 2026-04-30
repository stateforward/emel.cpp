---
phase: 114
status: passed
verified: 2026-04-27
requirements:
  - SPEECH-01
  - TOK-01
  - TOK-02
  - POLICY-01
  - PARITY-01
---

# Phase 114 Verification

## Requirement Verification

| Requirement | Result | Evidence |
|-------------|--------|----------|
| SPEECH-01 | passed | Maintained runtime is speech encoder/decoder/tokenizer; no top-level Whisper runtime or generic recognizer leak. |
| TOK-01 | passed | Compare and benchmark wrappers enforce tokenizer SHA `dfc530298b6fbed1a97c6472c575b026453706e2a204c7f7038f2c9d208b0759`. |
| TOK-02 | passed | The EMEL runner detokenizes decoder token ids through `speech/tokenizer/whisper/detail.hpp`. |
| POLICY-01 | passed | The EMEL runner uses `k_tiny_asr_decode_policy` for English transcribe/no-timestamps prompt roles. |
| PARITY-01 | passed | Compare summary records exact `[C]` parity on the pinned Phase 99 model/audio pair. |

## Command Evidence

```sh
python3 -m py_compile tools/bench/whisper_compare.py tools/bench/whisper_benchmark.py
```

Result: passed.

```sh
cmake --build build/whisper_compare_tools --target whisper_emel_parity_runner \
  whisper_benchmark_tests -j 4
build/whisper_compare_tools/whisper_benchmark_tests
```

Result: 7 test cases and 100 assertions passed.

```sh
build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/speech/tokenizer/*'
build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/speech/encoder/whisper/*'
build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/speech/decoder/whisper/*'
```

Result: tokenizer 3/3 tests and 26 assertions passed; encoder 13/13 tests and 1806 assertions
passed; decoder 4/4 tests and 1420 assertions passed.

```sh
scripts/check_domain_boundaries.sh
scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build
EMEL_WHISPER_BENCH_WARMUPS=0 EMEL_WHISPER_BENCH_ITERATIONS=1 \
  scripts/bench_whisper_single_thread.sh --skip-reference-build --skip-emel-build
```

Result: domain check passed; compare reported `exact_match`; benchmark reported
`benchmark_status=ok reason=ok`.

## Source-Backed Artifact Checks

- `build/whisper_compare/summary.json` EMEL record has
  `backend_id: emel.speech.whisper.encoder_decoder`.
- `build/whisper_compare/summary.json` EMEL record has
  `runtime_surface: speech/encoder/whisper+speech/decoder/whisper+speech/tokenizer/whisper`.
- `build/whisper_benchmark/benchmark_summary.json` records the same backend id and runtime
  surface.
