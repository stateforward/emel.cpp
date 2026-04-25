---
requirements-completed: []
---

# Phase 84 Summary: Speaker Activity And Segment Output Contract

## Outcome

Phase 84 is complete. Sortformer hidden frames now have a deterministic output contract for
speaker-activity probabilities and bounded segment records.

## Completed

- Added `src/emel/diarization/sortformer/output/detail.hpp`.
- Added `src/emel/diarization/sortformer/output/detail.cpp`.
- Added fixed-profile constants for `188` frames, `4` speakers, and 0.08-second frame shifts.
- Added `compute_speaker_probabilities` using maintained `mods.sh2s.*` tensors.
- Added `decode_segments` with independent per-speaker decoding, so overlapping speaker activity
  is explicit and allowed.
- Added stable speaker-label mapping for `speaker_0` through `speaker_3`.
- Added focused output lifecycle tests and wired them into the diarization shard.

## Verification

- `git diff --check -- CMakeLists.txt src/emel/diarization/sortformer/output/detail.hpp src/emel/diarization/sortformer/output/detail.cpp tests/diarization/sortformer/output/lifecycle_tests.cpp .planning/phases/84-speaker-activity-and-segment-output-contract`
- `cmake --build build/coverage --target emel_tests_bin -j 8`
- `ctest --test-dir build/coverage --output-on-failure -R '^emel_tests_diarization$' -j 1`
- `scripts/quality_gates.sh`

## Results

- Focused diarization shard after final label update: passed in `29.42s`.
- Final full quality gate: passed with exit `0`.
- Changed-source coverage from the final full gate:
  - lines: `96.3%`
  - branches: `68.2%`

## Notes

- The final quality gate printed a benchmark regression for `kernel/aarch64/op_soft_max`, then
  printed `warning: benchmark snapshot regression ignored by quality gates`.

## Deferred

- Lane-isolated parity proof and maintained ARM benchmark publication remain Phase 85.
