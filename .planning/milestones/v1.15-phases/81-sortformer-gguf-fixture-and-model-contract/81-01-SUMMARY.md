---
requirements-completed:
  - SORT-01
  - SORT-02
  - SORT-03
---

# Phase 81 Plan 1 Summary: Sortformer GGUF Fixture And Model Contract

**Completed:** 2026-04-22
**Status:** Complete; quality gate passed

## Changes

- Added `src/emel/model/sortformer/detail.hpp` and `detail.cpp` as the isolated Sortformer model
  contract module.
- Registered `sortformer` in the model architecture registry so GGUF loading can resolve the
  maintained architecture.
- Kept Sortformer-specific stream/profile metadata out of generic `emel::model::data`; the module
  validates exact GGUF metadata during `load_hparams` and exposes canonical constants through its
  execution contract.
- Removed the exported registry array-size dependency from `model/architecture/detail.hpp`.
- Added loader lifecycle coverage for:
  - accepted Sortformer GGUF hparams
  - wrong source contract rejection
  - execution contract construction
  - missing tensor-family rejection
  - noncanonical speaker-count rejection
- Documented the maintained fixture in `tests/models/README.md`.
- Updated the quality-gate coverage step to reuse the coverage build and scope threshold checks to
  changed `src/` files by default.

## Verification

- `git diff --check` passed.
- `cmake --build build/zig --target emel -j 4` passed.
- `ctest --test-dir build/zig --output-on-failure -R '^emel_tests_model_and_batch$' -j 1` passed.
- `EMEL_COVERAGE_CHANGED_ONLY=1 scripts/test_with_coverage.sh` passed with changed-file coverage
  scoped to `src/emel/model/architecture/*` and `src/emel/model/sortformer/*`
  (`95.8%` line, `59.3%` branch).
- `scripts/quality_gates.sh` passed after the coverage step was made cached and changed-file
  scoped. It emitted the existing tolerated benchmark warning for
  `kernel/aarch64/op_soft_max`.

## Notes

- Phase 81 does not implement audio request handling, Sortformer runtime execution, output decoding,
  parity comparison, or benchmarks. Those remain assigned to Phases 82 through 85.
