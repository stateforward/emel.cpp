---
phase: 144
status: passed
superseded-by: 146
---

# Phase 144 Validation

## Passing Evidence

- `cmake --build build/debug --target emel_tests_bin -j 6` passed.
- `build/debug/emel_tests_bin --test-case="generator_generate_quantized_contract_fixture_preserves_zero_disallowed_fallback" --no-skipped-summary` passed.
- `build/debug/emel_tests_bin --test-case="generator_generate_quantized_contract_fixture_supports_explicit_preselected_argmax_mode" --no-skipped-summary` passed.
- `ctest --test-dir build/debug -R emel_tests_generator_and_runtime --output-on-failure` passed.
- `rg -n "run_kernel_mode|plan->kind == step_kind::prefill \\?|detail::run_kernel_flash>|detail::run_kernel_nonflash>|detail::run_kernel_flash_preselected_argmax>|detail::run_kernel_nonflash_preselected_argmax>" src/emel/text/generator` returned no source hits.
- `scripts/check_domain_boundaries.sh` passed.
- `git diff --check` passed.
- `ctest --test-dir build/paritychecker_zig -R paritychecker_tests --output-on-failure` passed.

## Gate Issue

The scoped quality gate command:

```sh
EMEL_QUALITY_GATES_CHANGED_FILES='src/emel/text/generator/detail.hpp src/emel/text/generator/actions.hpp src/emel/text/generator/guards.hpp src/emel/text/generator/sm.hpp src/emel/text/generator/prefill/actions.hpp src/emel/text/generator/prefill/guards.hpp src/emel/text/generator/prefill/sm.hpp tests/text/generator/detail_tests.cpp tests/text/generator/action_guard_tests.cpp tests/text/generator/lifecycle_tests.cpp' EMEL_QUALITY_GATES_BENCH_SUITE=generation scripts/quality_gates.sh
```

compiled the zig and coverage generator shards and the coverage
`emel_tests_generator_and_runtime` run passed, but the coverage report then
failed with `failed minimum line coverage (got 0.0%, minimum 90.0%)` and a
0-line report. This historical result is not recorded as Phase 144 standalone
closeout evidence. Phase 145 repaired the changed-file coverage gap, and
Phase 146 passed the final scoped generation quality gate for the reopened
milestone.
