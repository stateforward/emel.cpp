---
requirements-completed:
  - DIA-01
  - DIA-02
  - RUN-01
  - RUN-03
  - OUT-01
  - OUT-02
  - OUT-03
---

# Phase 89 Summary: Maintained Sortformer E2E Runtime Orchestrator

## Completed

- Added `src/emel/diarization/sortformer/pipeline/` as a maintained Boost.SML E2E actor.
- Added `event::run` with reference-based contract, PCM, probability output, segment output,
  output counts, and error output fields.
- Wired the pipeline through:
  - `DiarizationRequest` feature extraction
  - native encoder-frame computation from features
  - `SortformerExecutor` hidden-frame execution
  - `compute_speaker_probabilities`
  - bounded `decode_segments`
- Added fixed actor-owned scratch buffers for features, encoder frames, hidden frames, encoder
  workspace, and bound contracts.
- Added `emel::SortformerPipeline` as an additive top-level alias.
- Added focused lifecycle tests for valid PCM-to-segment execution, repeated deterministic output,
  invalid sample-rate rejection, and probability-output capacity rejection.

## Verification

- `git diff --check -- src/emel/diarization/sortformer/pipeline src/emel/machines.hpp CMakeLists.txt tests/diarization/sortformer/pipeline .planning/phases/89-maintained-sortformer-e2e-runtime-orchestrator` passed.
- `cmake --build build/coverage --target emel_tests_bin -j 6` passed.
- `ctest --test-dir build/coverage --output-on-failure -R '^emel_tests_diarization$' -j 1`
  passed after the tensor-guard review fix: `100% tests passed, 0 tests failed out of 1`.
- `scripts/quality_gates.sh` passed after the tensor-guard review fix.
  - Changed-source coverage: `95.0%` line, `68.1%` branch.
  - The benchmark step reported existing tolerated benchmark regressions and unsnapshotted
    diarization rows, then printed `warning: benchmark snapshot regression ignored by quality
    gates`.

## Notes

- Phase 89 intentionally does not update parity or benchmark publication; Phase 90 owns that.
- Phase 89 intentionally does not repair pre-existing request/executor action branching; Phase 91
  owns that governance repair.
