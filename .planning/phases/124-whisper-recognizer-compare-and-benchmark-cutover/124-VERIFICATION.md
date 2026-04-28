---
phase: 124
status: pass
verified: 2026-04-28
---

# Verification 124

## Source Checks

- `rg -n "encoder::whisper|decoder::whisper|decode_token_ids|speech/encoder/whisper|speech/decoder/whisper|emel\\.speech\\.whisper\\.encoder_decoder" tools/bench/whisper_emel_parity_runner.cpp`
  returned no matches.
- `rg -n "whisper|model/whisper|speech/tokenizer/whisper|speech/encoder/whisper|speech/decoder/whisper|model::whisper|tokenizer::whisper|encoder::whisper|decoder::whisper" src/emel/speech/recognizer tests/speech/recognizer`
  returned no matches.
- `rg -n "emel/whisper|namespace emel::whisper|kernel/whisper|kernel::whisper" src tests CMakeLists.txt`
  returned no matches.
- `scripts/check_domain_boundaries.sh` passed.

## Build And Tests

- `cmake --build build/audit-native --target emel_tests_bin -j 6` passed.
- `cmake --build build/whisper_compare_tools --target whisper_emel_parity_runner whisper_benchmark_tests -j 6` passed.
- `build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/speech/recognizer/*'`
  passed: 7 test cases / 125 assertions.
- `build/audit-native/emel_tests_bin --no-breaks --test-case='whisper_recognizer*'`
  passed: 1 test case / 356 assertions.
- `build/whisper_compare_tools/whisper_benchmark_tests --no-breaks`
  passed: 10 test cases / 148 assertions.

## Maintained Evidence

- `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build --output-dir build/whisper_compare_phase124`
  passed with `status=exact_match reason=ok`.
- `scripts/bench_whisper_single_thread.sh --skip-reference-build --skip-emel-build --output-dir build/whisper_benchmark_phase124 --warmups 1 --iterations 3`
  passed with `benchmark_status=ok reason=ok`; that run recorded EMEL mean
  `58,238,902 ns` versus reference mean `62,824,389 ns`.
- The changed-file quality gate reran the default maintained outputs and passed:
  - `build/whisper_compare/summary.json`: `exact_match`, `ok`,
    `emel.speech.recognizer.whisper`,
    `speech/recognizer+speech/recognizer_routes/whisper`.
  - `build/whisper_benchmark/benchmark_summary.json`: `ok`, `ok`,
    `emel.speech.recognizer.whisper`,
    `speech/recognizer+speech/recognizer_routes/whisper`, EMEL mean
    `58,263,986 ns`, reference mean `60,507,152 ns`.

## Quality Gate

Command:

```bash
EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare:whisper_single_thread \
EMEL_QUALITY_GATES_CHANGED_FILES="src/emel/speech/recognizer/events.hpp:src/emel/speech/recognizer/actions.hpp:tests/speech/recognizer/lifecycle_tests.cpp:tools/bench/whisper_emel_parity_runner.cpp:tools/bench/whisper_compare.py:tools/bench/whisper_benchmark.py:tools/bench/whisper_benchmark_tests.cpp" \
scripts/quality_gates.sh
```

Result: passed. Coverage for changed recognizer headers was line `100.0%`. The gate skipped
irrelevant paritychecker and fuzz lanes, ran both selected Whisper proof suites, and skipped docs
generation because no docsgen-affecting files changed.

The gate printed existing Zig/locale warnings while configuring the `whisper.cpp` reference; they
did not fail the run.
