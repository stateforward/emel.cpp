---
phase: 127
status: passed
verified: 2026-04-28
requirements:
  - SPEECH-01
  - POLICY-01
  - CLOSE-01
---

# Phase 127 Verification

## Verdict

**Passed.** The maintained decoder actor no longer depends on encoder-owned detail code, and the
recognizer-backed Whisper compare and benchmark evidence still passes.

## Source Checks

- `rg -n "encoder/whisper/detail|encoder::whisper::detail" src/emel/speech/decoder/whisper`
  exited 1 with no matches.
- `rg -n "emel/whisper|namespace emel::whisper|kernel/whisper|kernel::whisper" src tests CMakeLists.txt`
  exited 1 with no matches.
- `scripts/check_domain_boundaries.sh` exited 0.
- `scripts/check_sml_behavior_selection.sh src/emel/speech/recognizer src/emel/speech/recognizer_routes/whisper src/emel/speech/encoder/whisper src/emel/speech/decoder/whisper src/emel/speech/tokenizer/whisper`
  exited 0.

## Build And Tests

- `cmake --build build/audit-native --target emel_tests_bin -j 6` passed.
- `build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/speech/decoder/whisper/*'`
  passed: 5 test cases, 1431 assertions.
- `build/audit-native/emel_tests_bin --no-breaks --test-case='*whisper_recognizer*'`
  passed: 1 test case, 356 assertions.
- `git diff --check` passed for Phase 127 files.

## Quality Gate

Command:

```bash
EMEL_QUALITY_GATES_CHANGED_FILES='src/emel/speech/decoder/whisper/actions.hpp:src/emel/speech/decoder/whisper/guards.hpp:src/emel/speech/decoder/whisper/detail.hpp:tests/speech/decoder/whisper/lifecycle_tests.cpp:.planning/phases/127-whisper-decoder-ownership-gap-closure/127-CONTEXT.md:.planning/phases/127-whisper-decoder-ownership-gap-closure/127-01-PLAN.md' \
EMEL_QUALITY_GATES_COVERAGE_CHANGED_FILES='src/emel/speech/decoder/whisper/actions.hpp:src/emel/speech/decoder/whisper/guards.hpp' \
EMEL_QUALITY_GATES_BENCH_SUITE='whisper_compare:whisper_single_thread' \
EMEL_WHISPER_BENCH_ITERATIONS=10 \
scripts/quality_gates.sh
```

Result:

- Build with zig: passed.
- Speech shard tests: passed.
- Changed-source coverage: line `98.5%`, branch `58.2%`.
- Paritychecker: skipped by changed-file scope.
- Fuzz smoke: skipped by changed-file scope.
- Whisper compare: `comparison_status=exact_match`, `reason=ok`, transcript `[C]`.
- Whisper single-thread benchmark: `status=ok`, `reason=ok`, EMEL mean `58,537,483 ns`,
  reference mean `60,435,595 ns`.

## Requirement Results

| Requirement | Result | Evidence |
|-------------|--------|----------|
| SPEECH-01 | passed | Decoder production files no longer include or alias encoder detail; domain checks passed. |
| POLICY-01 | passed | Timestamp-aware decoder policy execution now runs through decoder-owned detail from decoder actions. |
| CLOSE-01 | passed | Source-backed checks and scoped quality gate passed after the ownership repair. |
