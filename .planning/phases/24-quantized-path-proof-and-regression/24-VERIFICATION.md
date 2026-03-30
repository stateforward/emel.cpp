---
phase: 24-quantized-path-proof-and-regression
verified: 2026-03-25T21:16:08Z
status: passed
score: 3/3 phase truths verified
---

# Phase 24 Verification Report

**Phase Goal:** Extend maintained proof and regression surfaces so they fail if the canonical ARM
request regresses away from the approved quantized path.
**Verified:** 2026-03-25T21:16:08Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Maintained generation proof now publishes the shipped runtime contract directly and fails if the supported canonical path reports disallowed fallback, explicit no-claim, or a runtime-versus-audit mismatch. | ✓ VERIFIED | [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp) now emits `quantized_runtime_contract:` from the generator wrapper and rejects nonzero `disallowed_fallback` / `explicit_no_claim` counts plus runtime-audit drift. |
| 2 | The maintained `1/10/100/1000` parity surface now asserts the exact approved `8/4/0/0` contract instead of only checking for audit strings. | ✓ VERIFIED | [paritychecker_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/paritychecker_tests.cpp) now parses `quantized_runtime_contract:` and `quantized_stage_inventory:` lines and asserts `native_quantized=8`, `approved_dense_f32_by_contract=4`, `disallowed_fallback=0`, and `explicit_no_claim=0` across the maintained decode lengths. |
| 3 | Generator regression coverage now proves the supported canonical contract survives a real `generate` call, and the full repo gate remains green under current benchmark-warning policy. | ✓ VERIFIED | [lifecycle_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/generator/lifecycle_tests.cpp) adds `generator_generate_quantized_contract_fixture_preserves_zero_disallowed_fallback`, and `scripts/quality_gates.sh` completed successfully with only warning-only benchmark regressions. |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tools/paritychecker/parity_runner.cpp` | Runtime-contract publication and canonical regression-failure logic | ✓ EXISTS + SUBSTANTIVE | Publishes the shipped runtime contract and fails on supported-path fallback/no-claim or runtime-audit mismatch. |
| `tools/paritychecker/paritychecker_tests.cpp` | Exact maintained decode-length contract assertions | ✓ EXISTS + SUBSTANTIVE | Parses the new runtime contract line and asserts the exact supported contract across `1/10/100/1000`. |
| `tests/generator/lifecycle_tests.cpp` | Post-generate regression coverage for the supported contract | ✓ EXISTS + SUBSTANTIVE | Proves the quantized-contract fixture still reports `8/4/0/0` after an actual generate request. |
| `snapshots/quality_gates/timing.txt` | No leftover generated snapshot churn after full-gate verification | ✓ RESTORED | The full gate was run for Phase 24, then the generated timing snapshot was restored to the preserved baseline values. |

**Artifacts:** 4/4 verified

## Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| ATTR-01 | ✓ SATISFIED | - |
| VER-04 | ✓ SATISFIED | - |
| PAR-05 | ✓ SATISFIED | - |

## Gaps Summary

No remaining phase-local gaps.

Phase 25 is now the remaining milestone work: publish benchmark attribution that explains the
post-proof runtime contract honestly and isolates the next bottleneck.

Current non-blocking benchmark warnings remain:
- `batch/planner_simple`
- `memory/hybrid_full`
- `kernel/aarch64/op_log`

Those regressions were explicitly tolerated by the existing `quality_gates.sh` policy and should
be handled as benchmark-attribution work, not as a Phase 24 proof failure.

## Automated Checks

- `cmake --build build/paritychecker_zig --target paritychecker paritychecker_tests -j4`
- `./build/paritychecker_zig/paritychecker_tests`
- `./build/paritychecker_zig/paritychecker --generation --model /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1`
- `./build/paritychecker_zig/paritychecker --generation --model /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1000`
- `cmake --build build/zig --target emel_tests_bin -j4`
- `./build/zig/emel_tests_bin --test-case='*generator*quantized*' --no-breaks`
- `scripts/quality_gates.sh` ✓ exits `0`

## Verification Notes

- Phase 24 preserved the approved dense-f32-by-contract seams as visible contract stages rather
  than rewriting them into a misleading "fully quantized" claim.
- Unsupported-stage proof still belongs to the existing explicit-no-claim audit surface from
  Phase 22; Phase 24 did not collapse that negative path into a fabricated supported fallback case.
- Phase 24 closed without changing any SML transition table, actor ownership, or public C API
  boundary.

---
*Verified: 2026-03-25T21:16:08Z*
*Verifier: the agent*
