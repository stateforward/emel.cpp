---
phase: 40-native-q1-0-g128-runtime-bring-up
verified: 2026-04-02T21:05:29Z
status: passed
score: 3/3 phase truths verified
---

# Phase 40 Verification Report

**Phase Goal:** Add truthful EMEL-owned `Q1_0_g128` loading and native execution for the
maintained Bonsai slice on the shipped `qwen3` generator path.
**Verified:** 2026-04-02T21:05:29Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | EMEL now accepts the maintained Bonsai tensors as real upstream GGUF `Q1_0_g128` on the existing `qwen3` lane instead of colliding with project-owned pseudo-dtypes. | ✓ VERIFIED | [events.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/src/emel/kernel/events.hpp), [detail.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/src/emel/gguf/loader/detail.hpp), and [data.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/src/emel/model/data.cpp) now reserve upstream type `41` for real `Q1_0_g128` and move EMEL-owned packed/prepared dtypes into a non-upstream range. |
| 2 | `src/emel` now owns a native `Q1_0_g128` operand path for the maintained Bonsai slice, including layout/dequant support and `Q1_0_g128 x Q8_0` execution without a whole-tensor dequantize-to-f32 fallback in the hot path. | ✓ VERIFIED | [detail.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/src/emel/kernel/detail.hpp) now defines the block layout, row dequant helper, scalar dot product, and native `mul_mat` / `mul_mat_argmax` routing for `Q1_0_g128`. |
| 3 | The shipped generator path can initialize and produce bounded generation on a `qwen3` model carrying real `Q1_0_g128` logits weights under the maintained Bonsai formatter contract. | ✓ VERIFIED | [detail.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/src/emel/generator/detail.hpp) now dequantizes `Q1_0_g128` tensor rows correctly, and [lifecycle_tests.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/tests/generator/lifecycle_tests.cpp) proves initialization and one-token generation through a synthetic `qwen3` model using real `Q1_0_g128` logits weights. |

**Score:** 3/3 phase truths verified

## Automated Checks

- `cmake --build build/zig --parallel --target emel_tests_bin`
- `./build/zig/emel_tests_bin --test-case='*q1_0_g128*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='*qwen3*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='*generator_detail_copy_tensor_row_dequantizes_q1_0_g128_blocks*' --no-breaks`
- `./scripts/quality_gates.sh`

## Current Results

- Passed:
  - `cmake --build build/zig --parallel --target emel_tests_bin`
  - `./build/zig/emel_tests_bin --test-case='*q1_0_g128*' --no-breaks`
  - `./build/zig/emel_tests_bin --test-case='*qwen3*' --no-breaks`
  - `./build/zig/emel_tests_bin --test-case='*generator_detail_copy_tensor_row_dequantizes_q1_0_g128_blocks*' --no-breaks`
  - `./scripts/quality_gates.sh`

## Verification Notes

- The maintained Bonsai GGUF probe now succeeds because loader layout validation recognizes type
  `41` as `Q1_0_g128` storage (`128` elements, `18` bytes per block).
- Native kernel coverage now includes both `mul_mat` and `mul_mat_argmax` on `Q1_0_g128` weights.
- `scripts/quality_gates.sh` now uses a default `1800s` outer timeout so the full gate can
  complete the benchmark and docs tail without being killed after successful benchmark execution.
