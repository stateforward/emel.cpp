---
phase: 145
plan: 01
status: complete
requirements-completed: []
superseded-by: 146
verification: passed
validation: passed
---

# Summary 145-01: Native Quantized Route Evidence Closure

## Status

Complete. The source and behavior gaps identified by the milestone audit were addressed, the
changed-file scoped coverage lane passes, and the full changed-file scoped quality gate passes
after the network fix.

## Implemented

- Added a source regression in `tests/text/generator/lifecycle_tests.cpp` that scans the actual
  `matmul_vector_native_quantized(...)` helper body for hidden packed-q8/q8-k route probing.
- Removed dispatch-time packed-q8/q8-k probing from `matmul_vector_native_quantized(...)`.
- Replaced `phase_lifecycle(...)` runtime-indexed manifest selection with explicit
  `prefill_lifecycle(...)` and `decode_lifecycle(...)` helpers.
- Added explicit SML materialized-logits routes for native quantized body matmul with q8-k logits:
  - parent decode flash/nonflash routes
  - prefill flash/nonflash routes
- Preserved the quantized contract generation fixture by keeping q2/q3 body matmuls on the generic
  native quantized kernel path while routing q6 logits through q8-k logits from explicit guards.
- Added coverage tests for:
  - explicit run-kernel route wrapper rejection branches
  - route template rejection branches
  - prepared Qwen3 nonflash/flash scalar kernel and native-quantized paths
  - action/guard route classification and lifecycle capture branches

## Validation

Changed-file scoped quality gate now passes:

- line coverage: 92.7%, required 90.0%
- branch coverage: 50.2%, required 50.0%
- `emel_tests_generator_and_runtime`: passed under coverage
- `paritychecker_tests`: passed
- benchmark snapshot lane: passed
