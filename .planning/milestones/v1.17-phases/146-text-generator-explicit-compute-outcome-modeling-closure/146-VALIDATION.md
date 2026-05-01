---
phase: 146
status: passed
validated_at: "2026-04-30T04:39:58Z"
---

# Phase 146 Validation

## Focused Checks

- `cmake --build build/debug --target emel_tests_bin` passed.
- `./build/debug/emel_tests_bin --test-case="generator_scalar_kernel_route_choice_stays_in_state_machines" --no-skipped-summary` passed.
- `./build/debug/emel_tests_bin --test-case="generator_detail_run_kernel_callbacks_do_not_write_error_channel" --no-skipped-summary` passed.
- `./build/debug/emel_tests_bin --test-case="generator_detail_decode_preconditions_reject_malformed_requests" --no-skipped-summary` passed.
- `./build/debug/emel_tests_bin --test-case="generator compute readiness guards classify request and backend gaps" --no-skipped-summary` passed.
- `./build/debug/emel_tests_bin --test-case="generator guard detail predicates cover negative branch cases" --no-skipped-summary` passed.

## Integration Checks

- `ctest --test-dir build/debug -R "emel_tests_generator_and_runtime|emel_tests_text_runtime" --output-on-failure` passed.
- `scripts/check_domain_boundaries.sh` passed.
- `ctest --test-dir build/paritychecker_zig -R paritychecker_tests --output-on-failure` passed.
- `./build/bench_tools_ninja/bench_runner_tests --test-case="*generation*" --no-skipped-summary` passed.

## Quality Gate

Command:

```bash
EMEL_QUALITY_GATES_CHANGED_FILES="src/emel/text/generator/detail.hpp,src/emel/text/generator/guards.hpp,src/emel/text/generator/prefill/guards.hpp,src/emel/text/generator/sm.hpp,src/emel/text/generator/prefill/sm.hpp,tests/text/generator/detail_tests.cpp,tests/text/generator/action_guard_tests.cpp,tests/text/generator/lifecycle_tests.cpp,.planning/phases/146-text-generator-explicit-compute-outcome-modeling-closure/146-CONTEXT.md,.planning/phases/146-text-generator-explicit-compute-outcome-modeling-closure/146-01-PLAN.md" EMEL_QUALITY_GATES_BENCH_SUITE=generation scripts/quality_gates.sh
```

Result: passed.

- Generator/runtime shard passed.
- Changed-source coverage passed: line `92.5%`, branch `50.0%`.
- Paritychecker passed.
- Fuzz was skipped by the gate because no fuzz-affecting files changed.
- Generation benchmark comparison passed.
- Docs generation passed.

Earlier scoped gate attempts failed branch coverage while adding tests; those attempts were not used
as closeout evidence. The final command above is the accepted validation run.
