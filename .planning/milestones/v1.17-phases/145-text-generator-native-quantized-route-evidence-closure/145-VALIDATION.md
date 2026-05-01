---
phase: 145
status: passed
superseded-by: 146
---

# Validation 145: Native Quantized Route Evidence Closure

## Passing Evidence

- `build/debug/emel_tests_bin --test-case="generator_scalar_kernel_route_choice_stays_in_state_machines" --no-skipped-summary`
  - Passed after failing before the production fix.
- `build/debug/emel_tests_bin --test-case="generator_lfm2_quantized_path_audit_accepts_hybrid_native_quantized_mix" --no-skipped-summary`
  - Passed.
- `build/debug/emel_tests_bin --test-case="generator_initialize_quantized_contract_fixture_reports_zero_disallowed_fallback_stages" --no-skipped-summary`
  - Passed.
- `build/debug/emel_tests_bin --test-case="generator_generate_quantized_contract_fixture_preserves_zero_disallowed_fallback" --no-skipped-summary`
  - Passed after adding the explicit native-quantized q8-logits route.
- `build/debug/emel_tests_bin --test-case="generator_generate_quantized_contract_fixture_supports_explicit_preselected_argmax_mode" --no-skipped-summary`
  - Passed.
- `ctest --test-dir build/debug -R emel_tests_generator_and_runtime --output-on-failure`
  - Passed after added coverage tests: 1/1, 107.62 seconds.
- `build/debug/emel_tests_bin --test-case="generator_detail_run_kernel_route_wrappers_reject_missing_compute_io"`
  - Passed: 638 assertions.
- `build/debug/emel_tests_bin --test-case="generator_detail_route_templates_reject_unprepared_inputs"`
  - Passed: 56 assertions.
- `build/debug/emel_tests_bin --test-case="generator_detail_scalar_routes_run_prepared_qwen3_paths"`
  - Passed: 40 assertions.
- `ctest --test-dir build/paritychecker_zig -R paritychecker_tests --output-on-failure`
  - Passed: 1/1, 8.10 seconds.
- `scripts/check_domain_boundaries.sh`
  - Passed.
- `git diff --check`
  - Passed.
- Changed-file scoped quality gate coverage lane:
  - `emel_tests_generator_and_runtime` passed under coverage: 1/1, 119.40 seconds.
  - line coverage: 92.7% against required 90.0%
  - branch coverage: 50.2% against required 50.0%
- Full changed-file scoped quality gate:
  - Passed after network recovery.
  - `emel_tests_generator_and_runtime` passed: 1/1, 119.50 seconds.
  - coverage passed: 92.7% line, 50.2% branch.
  - `paritychecker_tests` passed: 1/1, 8.64 seconds.
  - `fuzz_smoke` skipped because no fuzz-affecting files changed.
  - benchmark snapshot lane passed and printed generation benchmark evidence.

## Historical Failed Evidence

Command:

```bash
EMEL_QUALITY_GATES_CHANGED_FILES='src/emel/text/generator/detail.hpp:src/emel/text/generator/actions.hpp:src/emel/text/generator/guards.hpp:src/emel/text/generator/sm.hpp:src/emel/text/generator/prefill/actions.hpp:src/emel/text/generator/prefill/guards.hpp:src/emel/text/generator/prefill/sm.hpp:tests/text/generator/lifecycle_tests.cpp' scripts/quality_gates.sh
```

Earlier result, before adding branch coverage tests:

- `emel_tests_generator_and_runtime` passed under coverage: 1/1, 117.51 seconds.
- Coverage threshold failed:
  - line coverage 87.2% vs required 90.0%
  - branch coverage 37.4% vs required 50.0%

Current result, after adding tests:

- Coverage thresholds pass:
  - line coverage 92.7% vs required 90.0%
  - branch coverage 50.2% vs required 50.0%
- Full `scripts/quality_gates.sh` now passes after the network fix.

## Verification Notes

- `src/emel/text/generator/detail.hpp` no longer defines `phase_lifecycle(...)`.
- `matmul_vector_native_quantized(...)` no longer calls
  `packed_q8_0_input_path_supported(...)`, `q8_input_path_supported(...)`,
  `matmul_vector_packed_q8_0(...)`, or `matmul_vector_q8_k(...)`.
- The previous coverage threshold blocker is closed by added tests in
  `tests/text/generator/detail_tests.cpp` and `tests/text/generator/action_guard_tests.cpp`.
- The previous external dependency fetch caveat is closed by the passing full gate rerun.
