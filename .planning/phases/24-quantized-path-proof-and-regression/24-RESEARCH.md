# Phase 24: Quantized Path Proof And Regression - Research

**Researched:** 2026-03-25
**Domain:** Maintained quantized-path proof and regression hardening
**Confidence:** HIGH

## Summary

The smallest truthful Phase 24 gap is not in the shipped runtime anymore; it is in the maintained
proof surfaces. Phase 23 already proved the supported canonical initialized runtime reports
`native_quantized=8`, `approved_dense_f32_by_contract=4`, `disallowed_fallback=0`, and
`explicit_no_claim=0`. But `tools/paritychecker --generation` still recomputes the quantized-path
audit from `model_data` locally and only prints the result. It does not yet consume the shipped
runtime contract counts or fail if the canonical supported path regresses away from the approved
contract.

The test surface has the same gap. Current paritychecker tests iterate the maintained
`1/10/100/1000` decode lengths and check that audit strings exist, but they do not assert the
exact `8/4/0/0` contract or any hard-fail regression behavior. Generator tests already prove the
runtime contract at initialization time, and Phase 22 already proves unsupported `q4_0` stays
`explicit_no_claim`. So Phase 24 should harden maintained proof, not reopen the runtime contract.

## Evidence

- `src/emel/generator/sm.hpp:798-820`
  The shipped generator wrapper now exposes `generation_native_quantized_stage_count()`,
  `generation_approved_dense_f32_stage_count()`,
  `generation_disallowed_fallback_stage_count()`, and
  `generation_explicit_no_claim_stage_count()`.
- `tests/generator/lifecycle_tests.cpp:669-681`
  `generator_initialize_quantized_contract_fixture_reports_zero_disallowed_fallback_stages`
  already proves the initialized supported runtime reports `8/4/0/0`.
- `tools/paritychecker/parity_runner.cpp:15233-15278`
  Generation mode currently rebuilds the audit from `build_execution_view(*state.model_data, ...)`
  and prints `quantized_stage_inventory:` plus `quantized_stage_audit:` rows, but it does not yet
  fail on disallowed fallback from the shipped runtime contract.
- `tools/paritychecker/paritychecker_tests.cpp:468-476`
  `check_generation_quantized_stage_audit(...)` only checks that the audit strings are present.
- `tools/paritychecker/paritychecker_tests.cpp:523-551`
  The maintained `1/10/100/1000` parity loop already exists and is the right place to assert the
  exact contract counts and regression-failure semantics.

## Recommended Plan Split

### 24-01: Promote The Runtime Contract Into Paritychecker Proof

- Teach `tools/paritychecker --generation` to consume the shipped generator runtime contract counts
  and publish them explicitly on the maintained output surface.
- Fail canonical generation mode if the supported path reports any `disallowed_fallback` or
  `explicit_no_claim` stages, and optionally sanity-check the runtime counts against the model
  audit if that helps keep publication honest.
- Extend paritychecker tests so the maintained `1/10/100/1000` workloads assert the exact
  supported `8/4/0/0` contract instead of only string presence.

### 24-02: Add Regression Coverage Around The Approved Contract

- Add the narrowest generator/parity regression checks needed so supported-path proof stays locked
  to `8/4/0/0` while unsupported-stage proof remains explicit `no-claim`.
- Reuse the existing Phase 22 unsupported negative path rather than fabricating a bogus supported
  fallback fixture.
- Finish by running the full repo gate so the hardened proof surface lands without snapshot churn.

## User Approval Check

No user approval is needed for the recommended Phase 24 work if it stays inside paritychecker,
generator wrappers/tests, and other additive proof surfaces. I do **not** recommend changing SML
transition tables, actor ownership, or benchmark/docs publication scope here.
