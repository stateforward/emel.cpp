---
phase: 129
status: passed
verified: 2026-04-28
requirements: []
---

# Phase 129 Verification

## Verdict

**Passed.** Encoder detail no longer contains duplicate decoder/timestamp runtime helpers, and
the maintained recognizer-backed Whisper compare and benchmark path still passes.

## Source Checks

- Encoder detail helper-removal grep exited 0 through an inverted check, proving no matches for
  `decode_policy_runtime`, `required_decoder_workspace_floats`,
  `compute_decoder_cross_cache`, `run_decoder_layer_sequence`,
  `compute_decoder_logits_for_tokens`, `select_greedy_timestamp_aware_token`,
  `run_decoder_sequence`, `k_decoder`, `k_vocab_size`, or `k_token_`.
- Decoder production ownership grep exited 0 through an inverted check, proving no matches for
  `encoder/whisper/detail` or `encoder::whisper::detail` under
  `src/emel/speech/decoder/whisper`.
- Forbidden model-family root grep over `src`, `tests`, and `CMakeLists.txt` exited 0 through an
  inverted check.

## Build And Tests

- `cmake -S . -B build/audit-native -G Ninja` passed.
- `cmake --build build/audit-native --target emel_tests_bin -j 6` passed.
- `build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/speech/encoder/whisper/*'`
  passed: 15 test cases, 2166 assertions.
- `build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/speech/decoder/whisper/*'`
  passed: 7 test cases, 1436 assertions.
- `ctest --test-dir build/audit-native -R '^(emel_tests_speech|emel_tests_whisper)$' --output-on-failure`
  passed.

## Integration Checks

- `scripts/check_sml_behavior_selection.sh src/emel/speech/recognizer src/emel/speech/recognizer_routes/whisper src/emel/speech/encoder/whisper src/emel/speech/decoder/whisper src/emel/speech/tokenizer/whisper`
  passed.
- `scripts/check_domain_boundaries.sh` passed.
- `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` passed with
  `status=exact_match reason=ok`.
- `scripts/bench_whisper_single_thread.sh --skip-reference-build --skip-emel-build` passed with
  `benchmark_status=ok reason=ok`.
- Changed-file scoped `scripts/quality_gates.sh` passed with speech coverage, Whisper compare,
  and Whisper single-thread benchmark.

## Quality Notes

- Encoder-detail changed-file coverage: line `100.0%`, branch `50.0%`.
- After explicit user approval, `scripts/lint_snapshot.sh --update` refreshed
  `snapshots/lint/clang_format.txt`.
- `ctest --test-dir build/audit-native -R '^lint_snapshot$' --output-on-failure` passed.

## Requirement Results

| Requirement | Result | Evidence |
|-------------|--------|----------|
| Active v1.16 requirements | unchanged | Phase 129 is tech-debt cleanup only; active requirement owners remain Phases 123, 124, and 127. |
| Helper deduplication | passed | Encoder detail no longer contains duplicate decoder/timestamp helpers. |
| Decoder ownership | passed | Decoder production files still do not include or alias encoder detail. |
| Maintained behavior | passed | Recognizer-backed compare and benchmark still pass with exact `[C]` transcripts. |
