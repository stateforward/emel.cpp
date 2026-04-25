---
phase: 81
slug: sortformer-gguf-fixture-and-model-contract
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-22
---

# Phase 81 Validation: Sortformer GGUF Fixture And Model Contract

**Validated:** 2026-04-22

## Automated Checks

| Check | Result | Notes |
|-------|--------|-------|
| `git diff --check` | PASS | No whitespace errors. |
| `cmake --build build/zig --target emel -j 4` | PASS | Sortformer model module and architecture registry compile. |
| `ctest --test-dir build/zig --output-on-failure -R '^emel_tests_model_and_batch$' -j 1` | PASS | Model/loader shard passed with the new Sortformer tests. |
| `EMEL_COVERAGE_CHANGED_ONLY=1 scripts/test_with_coverage.sh` | PASS | Changed-file coverage scoped to the modified `src/` files passed at `95.8%` line and `59.3%` branch. |
| `scripts/quality_gates.sh` | PASS | Passed with cached changed-file coverage; emitted the existing tolerated benchmark warning for `kernel/aarch64/op_soft_max`. |

## Requirement Coverage

| Requirement | Evidence | Status |
|-------------|----------|--------|
| SORT-01 | `tests/models/README.md` documents source URL, maintained path, license, upstream model, repo commit, size, linked ETag, Xet hash, download URL, metadata truth, stream contract, and conversion source. | Covered |
| SORT-02 | `src/emel/model/sortformer/detail.cpp` validates exact source format, tensor-name scheme, outtype, stream profile, speaker count, and tensor families; lifecycle tests cover accept/reject behavior. | Covered |
| SORT-03 | `sortformer::detail::execution_contract` names maintained stream parameters and required tensor families for later runtime execution. | Covered |

## Residual Risk

Phase 81 does not execute Sortformer audio inference. Runtime, output decoding, parity proof, and
benchmark publication remain assigned to Phases 82 through 85.
