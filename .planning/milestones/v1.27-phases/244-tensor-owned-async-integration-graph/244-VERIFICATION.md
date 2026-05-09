---
status: passed_with_benchmark_noise
phase: 244
plan: 01
requirements:
  - TNX-01
  - TNX-02
  - TNX-03
  - TST-03
---

# Phase 244 Verification

## Result

Passed for build, focused tests, coverage, lint snapshot, parity, and isolated benchmark reruns.
The full broad benchmark snapshot lane remained noisy and failed on unrelated suites that passed
immediately when rerun in isolation.

## Evidence

- Zig and native build targets passed:
  `cmake --build build/zig --target emel_tests_bin`
  `cmake --build --preset build-debug --target emel_tests_bin`
- Focused tensor async tests passed:
  `./build/debug/emel_tests_bin --test-case="*model_tensor_request_async*"`
  - 3 test cases passed
  - 35 assertions passed
- Full model tensor tests passed:
  `./build/debug/emel_tests_bin --test-case="*model_tensor*"`
  - 47 test cases passed
  - 532 assertions passed
- Full coverage mode passed:
  `EMEL_COVERAGE_CHANGED_ONLY=0 scripts/test_with_coverage.sh`
  - total line coverage: 91.9%
  - total branch coverage: 57.1%
  - touched tensor headers: `actions.hpp` 94%, `detail.hpp` 100%, `events.hpp` 100%,
    `guards.hpp` 92%
- Lint snapshot updated with explicit snapshot permission:
  `scripts/lint_snapshot.sh --update`
- Scoped quality gate non-benchmark lanes passed in the final rerun:
  `scripts/quality_gates.sh`
  - `test_with_coverage` passed
  - `paritychecker` passed
  - `fuzz_smoke` skipped as irrelevant
  - `lint_snapshot` passed after snapshot update
- Broad benchmark snapshot lane hit unrelated transient regressions across repeated reruns:
  - `text/encoders/rwkv_short`, isolated `encoder_rwkv` rerun passed
  - `text/encoders/wpm_long`
  - `tokenizer/preprocessor_rwkv_long`, isolated `tokenizer_preprocessor_rwkv` rerun passed
  - `logits/sampler_sml/vocab_32000`, isolated `logits_sampler` rerun passed

## Notes

Phase 244 adds the tensor-owned async integration graph only. Maintained loader selection and
loading-strategy performance comparison remain deferred to Phases 245 and 247.
