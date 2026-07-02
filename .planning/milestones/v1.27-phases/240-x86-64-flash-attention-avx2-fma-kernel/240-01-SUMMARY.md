---
phase: 240
status: passed
requirements-completed:
  - XFL-01
  - XFL-02
requirements-blocked: []
verification: passed
---

# Phase 240 Summary

## What Changed

- Added an EMEL-owned x86_64 one-chunk flash-attention kernel using AVX2/FMA
  f32 vector arithmetic and F16C f16 conversions.
- Kept the flash operand contract aligned with the AArch64 optimized path: f32
  Q rounded to f16, f16 K/V operands, f16 workspace accumulation, and f32 output.
- Added explicit optimized/shared x86_64 flash dispatch counters and actor
  accessors for attribution.
- Routed `op_flash_attn_ext` through optimized x86_64 guards/transitions before
  shared fallback and invalid handling.
- Added x86_64 tests proving optimized route, shared fallback when the feature
  contract is disabled, workspace reuse, and numeric agreement with maintained
  flash reference helpers.

## Validation

- Failing-first x86_64 test object compile: red captured before implementation.
- x86_64 test object compile: pass after implementation.
- `emel_tests_bin` build: pass.
- `emel_tests_kernel_and_graph` CTest shard: pass.
- Unsupported x86 feature flag source scan: pass.
- `scripts/lint_snapshot.sh`: pass without snapshot updates.
- Scoped `scripts/quality_gates.sh`: coverage, paritychecker, benchmark
  snapshot, lint, docs, and fuzz routing pass after approved snapshot updates.

## Closeout Status

The Phase 240 implementation satisfies and verifies `XFL-01` and `XFL-02`.
