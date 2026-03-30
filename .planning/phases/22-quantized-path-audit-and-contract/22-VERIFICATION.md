---
phase: 22-quantized-path-audit-and-contract
verified: 2026-03-25T20:15:49Z
status: passed
score: 3/3 phase truths verified
---

# Phase 22 Verification Report

**Phase Goal:** Inventory the maintained canonical ARM operand path and encode the supported versus
unsupported quantized-path contract explicitly.
**Verified:** 2026-03-25T20:15:49Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | The canonical ARM generation slice now has one shared audit source that classifies maintained stage families from the shipped runtime/model chain without actor rewrites. | ✓ VERIFIED | [detail.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/llama/detail.hpp) and [data.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/data.cpp) now expose `build_quantized_path_audit(...)`, stage-family names, contract names, and tensor-type names derived from the canonical Llama `execution_view`. |
| 2 | Unsupported or not-yet-ported quantized stage families now publish explicit no-claim behavior instead of silently inheriting approved dense-f32 or native-quantized claims. | ✓ VERIFIED | [lifecycle_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/generator/lifecycle_tests.cpp) adds `generator_quantized_path_audit_marks_unsupported_quantized_stage_no_claim`, proving an unsupported `q4_0` stage is reported as `explicit_no_claim`. |
| 3 | Maintained parity output now publishes stage inventory and per-stage audit rows alongside the existing q2/q3/q6 dispatch attribution, grounded in the shipped canonical runtime path. | ✓ VERIFIED | [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp) now prints `quantized_stage_inventory:` plus repeated `quantized_stage_audit:` lines, and direct `paritychecker --generation` runs at `max_tokens=1`, `1000`, and `--dump` all publish the audited contract with `supported=` and `consistent_across_layers=` fields. |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/emel/model/llama/detail.hpp` | Shared stage-audit types and helper declarations | ✓ EXISTS + SUBSTANTIVE | Defines the quantized stage families, contract kinds, audit records, and helper surface for the canonical Llama execution view. |
| `src/emel/model/data.cpp` | Canonical stage classification and no-claim logic | ✓ EXISTS + SUBSTANTIVE | Implements stage-family mapping, canonical contract classification, naming helpers, and unsupported-stage `explicit_no_claim` behavior. |
| `tests/generator/lifecycle_tests.cpp` | Generator-focused positive and negative audit coverage | ✓ EXISTS + SUBSTANTIVE | Proves canonical stage-family classification and unsupported quantized no-claim behavior. |
| `tools/paritychecker/parity_runner.cpp` | Maintained stage inventory and per-stage audit publication | ✓ EXISTS + SUBSTANTIVE | Prints quantized stage inventory and parseable audit rows next to the existing runtime dispatch metrics. |
| `tools/paritychecker/paritychecker_tests.cpp` | Maintained proof-surface assertions for the new audit output | ✓ EXISTS + SUBSTANTIVE | Checks that the generation output includes the new stage inventory and audit strings. |

**Artifacts:** 5/5 verified

## Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| AUD-01 | ✓ SATISFIED | - |
| PATH-02 | ✓ SATISFIED | - |

## Gaps Summary

No remaining phase-local gaps.

Phase 23 remains the next milestone question: determine whether any supported canonical branches
still qualify as disallowed fallback or whether the closure phase is purely proof of zero remaining
disallowed branches.

## Automated Checks

- `cmake --build build/zig --target emel_tests_bin -j4`
- `cmake --build build/paritychecker_zig --target paritychecker paritychecker_tests -j4`
- `./build/zig/emel_tests_bin --test-case='*generator*quantized*audit*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='*generator*no*claim*' --no-breaks`
- `rg -n 'no-claim|operator_inventory|stage_inventory|quantized.*audit|contract' src/emel/model/llama/detail.hpp src/emel/model/data.cpp tools/paritychecker/parity_runner.cpp tools/paritychecker/paritychecker_tests.cpp tests/generator/lifecycle_tests.cpp`
- `./build/paritychecker_zig/paritychecker --generation --model /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1`
- `./build/paritychecker_zig/paritychecker --generation --model /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1 --dump`
- `./build/paritychecker_zig/paritychecker --generation --model /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1000`
- `scripts/quality_gates.sh` ✓ exits `0`

## Verification Notes

- The canonical short-case audit now reconciles to `native_quantized=8`,
  `approved_dense_f32_by_contract=4`, `disallowed_fallback=0`, and `explicit_no_claim=0`.
- The rebuilt paritychecker binary now publishes the per-stage `supported=` field as intended.
- Phase 22 closed without changing any SML transition table, actor ownership, or public C API
  boundary.

---
*Verified: 2026-03-25T20:15:49Z*
*Verifier: the agent*
