---
phase: 69-generative-compare-contract
plan: 01
status: complete
completed: 2026-04-20
requirements-completed:
  - GEN-01
  - GEN-02
  - ISO-01
---

# Phase 69 Summary

## Outcome

Phase 69 is complete. The maintained generation bench now has one canonical
`generation_compare/v1` JSONL contract shared by the EMEL and C++ reference lanes, while keeping
the existing human-readable benchmark output intact.

## Delivered

- Added `tools/bench/generation_compare_contract.hpp` as the canonical schema/versioned JSONL
  emission surface for maintained generation compare records.
- Extended the maintained generation bench result surface so EMEL and reference runs record lane
  identity, workload metadata, formatter/sampling/stop metadata, checksums, and optional dumped
  outputs without widening `src/` runtime scope.
- Added JSONL-mode regression coverage in `tools/bench/bench_runner_tests.cpp` proving the shared
  schema is emitted for both lanes and that legacy text-mode output still works.

## Contract Truth

- The first shared schema is `generation_compare/v1`.
- EMEL and reference generation records stay lane-local:
  - `lane=emel`, `backend_id=emel.generator`
  - `lane=reference`, `backend_id=cpp.reference.llama_cpp`
- JSONL emission is additive. Operators can still use the pre-existing human-readable benchmark
  and compare output path.
- The phase stays inside `tools/bench` and tests. No EMEL runtime ownership moved into reference
  tooling and no reference runtime objects were introduced into `src/`.

## Verification Result

- Focused generation bench build and test commands passed.
- The full repo `scripts/quality_gates.sh` run passed after repairing an unrelated paritychecker
  debug-field drift from `llama_layer.bo` to `llama_layer.wo_b`.
- The milestone is now ready to plan and execute Phase `70` on top of a passing shared contract
  seam.
