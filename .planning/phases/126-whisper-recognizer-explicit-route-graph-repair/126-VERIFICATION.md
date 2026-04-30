---
phase: 126
status: passed
verified: 2026-04-28
---

# Verification 126

## Source-Backed Checks

- `src/emel/speech/recognizer/events.hpp` no longer defines `runtime_backend` or
  `initialize.backend`.
- `src/emel/speech/recognizer/context.hpp` no longer stores a backend pointer.
- `src/emel/speech/recognizer/sm.hpp` selects route readiness and route phase execution through
  route-policy guard/action types in explicit SML transition rows.
- `src/emel/speech/recognizer_routes/whisper/any.hpp` defines the Whisper route policy while the
  generic recognizer tree stays model-family-free.
- `tools/bench/whisper_emel_parity_runner.cpp` instantiates
  `speech_recognizer::sm<whisper_route::route>` and dispatches public initialize/recognize events.

## Focused Commands

- `cmake --build build/audit-native --target emel_tests_bin -j 6` passed.
- `build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/speech/recognizer/*'`
  passed: 7 test cases / 121 assertions.
- `build/audit-native/emel_tests_bin --no-breaks --test-case='*whisper_recognizer*'` passed:
  1 test case / 356 assertions.
- `ctest --test-dir build/audit-native -R 'emel_tests_(speech|whisper)$' --output-on-failure`
  passed: 2/2 tests.
- `cmake --build build/whisper_compare_tools -j 6` passed.
- `build/whisper_compare_tools/whisper_benchmark_tests --no-breaks` passed: 10 test cases /
  148 assertions.

## Rule And Boundary Commands

- `scripts/check_sml_behavior_selection.sh src/emel/speech/recognizer src/emel/speech/recognizer_routes/whisper src/emel/speech/encoder/whisper src/emel/speech/decoder/whisper src/emel/speech/tokenizer/whisper`
  passed.
- `scripts/check_domain_boundaries.sh` passed.
- `rg -n "runtime_backend|initialize_ev\\.backend|ctx\\.backend|recognizer_routes::whisper::backend|route::backend|ctx\\.backend->|backend->" src/emel/speech/recognizer src/emel/speech/recognizer_routes tests/speech tools/bench/whisper_emel_parity_runner.cpp`
  returned no matches.
- `rg -n "emel/whisper|namespace emel::whisper|kernel/whisper|kernel::whisper" src tests CMakeLists.txt`
  returned no matches.
- `rg -n "whisper" src/emel/speech/recognizer src/emel/speech/recognizer/events.hpp src/emel/speech/recognizer/context.hpp src/emel/speech/recognizer/guards.hpp src/emel/speech/recognizer/actions.hpp src/emel/speech/recognizer/sm.hpp`
  returned no matches.

## Maintained Evidence

- `build/whisper_compare/summary.json`: `comparison_status=exact_match`, `reason=ok`,
  backend `emel.speech.recognizer.whisper`, runtime surface
  `speech/recognizer+speech/recognizer_routes/whisper`, transcript `[C]`.
- `build/whisper_benchmark/benchmark_summary.json`: `status=ok`, `reason=ok`, backend
  `emel.speech.recognizer.whisper`, runtime surface
  `speech/recognizer+speech/recognizer_routes/whisper`, transcript `[C]`, EMEL mean
  `58,911,208 ns`, reference mean `60,982,694 ns`.

## Quality Gate

Changed-file scoped command:

Command:

```bash
EMEL_QUALITY_GATES_CHANGED_FILES='src/emel/speech/recognizer/actions.hpp:src/emel/speech/recognizer/context.hpp:src/emel/speech/recognizer/events.hpp:src/emel/speech/recognizer/guards.hpp:src/emel/speech/recognizer/sm.hpp:src/emel/speech/recognizer_routes/whisper/actions.hpp:src/emel/speech/recognizer_routes/whisper/any.hpp:src/emel/speech/recognizer_routes/whisper/detail.cpp:src/emel/speech/recognizer_routes/whisper/guards.hpp:tools/bench/whisper_emel_parity_runner.cpp:tests/speech/recognizer/lifecycle_tests.cpp:tests/speech/encoder/whisper/lifecycle_tests.cpp' \
EMEL_QUALITY_GATES_BENCH_SUITE='whisper_compare:whisper_single_thread' \
scripts/quality_gates.sh
```

Result: passed.

- Domain-boundary gate passed.
- Zig speech shard build passed.
- Changed-file coverage passed: line `96.9%`, branch `55.7%`.
- Paritychecker skipped as irrelevant to the changed files.
- Fuzz smoke skipped as irrelevant to the changed files.
- Whisper compare passed with `status=exact_match reason=ok`.
- Whisper single-thread benchmark passed with `benchmark_status=ok reason=ok`.
- Docs generation completed.

Full closeout command:

```bash
EMEL_QUALITY_GATES_SCOPE=full \
EMEL_QUALITY_GATES_BENCH_SUITE='whisper_compare:whisper_single_thread' \
scripts/quality_gates.sh
```

Result: passed.

- Domain-boundary gate passed.
- Zig build passed.
- Full coverage test set passed: 12/12 shards.
- Coverage passed: line `90.8%`, branch `55.6%`.
- Paritychecker tests passed.
- Fuzz smoke passed for GGUF parser, GBNF parser, Jinja parser, and Jinja formatter corpora.
- Whisper compare passed with `status=exact_match reason=ok`.
- Whisper single-thread benchmark passed with `benchmark_status=ok reason=ok`.
- Docs generation completed.

## Non-Blocking Snapshot Note

`ctest --test-dir build/audit-native -R '^lint_snapshot$' --output-on-failure` failed because the
existing `snapshots/lint/clang_format.txt` baseline is stale for unrelated Whisper
decoder/encoder files and would need explicit snapshot update approval. Phase 126 changed files
were formatted and the changed-file quality gate passed.
