---
requirements-completed: []
---

# Phase 83 Summary: Maintained Sortformer Contract Repair

## Outcome

Phase 83 is complete as the prerequisite contract-repair slice for native Sortformer execution.
It no longer claims to finish the whole runtime path.

## Completed

- Verified the maintained OpenResearchTools GGUF against a self-converted NVIDIA NeMo checkpoint.
- Repaired the maintained Sortformer profile constants to the real GGUF metadata.
- Updated tensor-family acceptance to `prep.`, `enc.`, `mods.`, and `te.`.
- Moved deterministic audio feature extraction into the stage-owned
  `src/emel/diarization/sortformer/encoder/feature_extractor/` component.
- Split the remaining RUN work into stage-owned decimal phases:
  - Phase 83.1: encoder binding and kernels
  - Phase 83.2: modules and speaker cache
  - Phase 83.3: transformer encoder path
  - Phase 83.4: execution orchestrator

## Verification

- `cmake --build build/coverage --target emel_tests_bin -j 8`
- `ctest --test-dir build/coverage --output-on-failure -R '^emel_tests_diarization$' -j 1`
- `ctest --test-dir build/coverage --output-on-failure -R '^emel_tests_model_and_batch$' -j 1`
- `EMEL_QUALITY_GATES_TIMEOUT=3600s scripts/quality_gates.sh`

## Notes

RUN-01 remains intentionally deferred to the decimal execution phases. The milestone must not accept
a synthetic projection, head-only path, generic runtime dispatcher, or external fallback as native
Sortformer execution.
