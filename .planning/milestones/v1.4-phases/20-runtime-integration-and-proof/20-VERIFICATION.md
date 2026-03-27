---
phase: 20-runtime-integration-and-proof
verified: 2026-03-25T18:51:10Z
status: passed
score: 4/4 phase truths verified
---

# Phase 20 Verification Report

**Phase Goal:** Adopt the complete vectorized quantized kernel set in the shipped runtime chain and
prove supported plus fallback behavior without changing public APIs or actor structure.
**Verified:** 2026-03-25T18:51:10Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | The shipped generator -> graph -> processor -> kernel chain can publish q2/q3/q6 optimized/shared runtime attribution without actor rewrites or public API widening. | ✓ VERIFIED | [any.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/any.hpp) now forwards q2/q3/q6 counts additively, and [sm.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/generator/sm.hpp) now exposes generation-time q2/q3/q6 accessors alongside the existing flash attribution surface. |
| 2 | The canonical quantized generation path publishes proof that supported AArch64 execution exercised optimized q2/q3/q6 dispatch with zero shared fallback claims. | ✓ VERIFIED | [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp) now prints `quantized_dispatch:` metrics and fails canonical AArch64 proof if q2/q3/q6 optimized counts are zero or shared counts are nonzero; direct `paritychecker --generation` output on the Q2_K fixture shows optimized q2/q3/q6 counts > 0 with shared counts = 0. |
| 3 | The maintained generation parity gate now covers the full `1/10/100/1000` decode-length surface while proving quantized runtime attribution on the canonical fixture. | ✓ VERIFIED | [paritychecker_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/paritychecker_tests.cpp) now enumerates decode lengths `{1, 10, 100, 1000}`; direct `paritychecker --generation` runs on 2026-03-25 returned `generation parity ok` at `max_tokens=100` and `max_tokens=1000`. |
| 4 | Non-quantized runtime paths do not make false optimized quantized claims. | ✓ VERIFIED | [lifecycle_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/generator/lifecycle_tests.cpp) adds `generator_generate_f32_fixture_does_not_claim_quantized_optimized_dispatch`, proving the maintained f32 unit fixture reports zero q2/q3/q6 optimized/shared dispatch claims. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/emel/kernel/any.hpp` | q2/q3/q6 runtime attribution forwarding | ✓ EXISTS + SUBSTANTIVE | Adds additive optimized/shared q2/q3/q6 accessors. |
| `src/emel/generator/sm.hpp` | generation-time q2/q3/q6 attribution accessors | ✓ EXISTS + SUBSTANTIVE | Exposes runtime attribution without public API changes. |
| `tests/generator/lifecycle_tests.cpp` | negative no-claim runtime coverage | ✓ EXISTS + SUBSTANTIVE | Adds f32 fixture proof for zero q2/q3/q6 claims. |
| `tools/paritychecker/parity_runner.cpp` | canonical q2/q3/q6 runtime publication | ✓ EXISTS + SUBSTANTIVE | Prints `quantized_dispatch:` metrics and validates canonical AArch64 optimized-path claims. |
| `tools/paritychecker/paritychecker_tests.cpp` | maintained parity gate and canonical attribution checks | ✓ EXISTS + SUBSTANTIVE | Enforces maintained `1/10/100/1000` parity lengths and q2/q3/q6 metric checks. |

**Artifacts:** 5/5 verified

## Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| ARCH-02 | ✓ SATISFIED | - |
| PAR-04 | ✓ SATISFIED | - |
| VER-03 | ✓ SATISFIED | - |

## Gaps Summary

No remaining phase-local gaps.

Residual repo policy debt remains benchmark-variance noise, but that no longer blocks Phase 20 or
the milestone closeout.

## Automated Checks

- `cmake --build build/zig --target emel_tests_bin -j4`
- `cmake --build build/paritychecker_zig --target paritychecker paritychecker_tests -j4`
- `./build/zig/emel_tests_bin --test-case='*generator_generate_f32_fixture_does_not_claim_quantized_optimized_dispatch*' --no-breaks`
- `ctest --test-dir build/paritychecker_zig_latest --output-on-failure -R paritychecker_tests`
- `./build/paritychecker_zig/paritychecker_tests --test-case='*generation dump proves the EMEL path avoids the reference decode seam*' --no-breaks`
- `./build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 10`
- `./build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 100`
- `./build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1000`
- `./build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1 --dump`

## Verification Notes

- Commit `15f2334` restored long-decode parity by returning the maintained runtime path to the
  exact masked nonflash attention semantics needed for stable `100/1000` decode agreement.
- Phase 20 runtime-chain attribution and full maintained parity proof are now green.

---
*Verified: 2026-03-25T18:51:10Z*
*Verifier: the agent*
