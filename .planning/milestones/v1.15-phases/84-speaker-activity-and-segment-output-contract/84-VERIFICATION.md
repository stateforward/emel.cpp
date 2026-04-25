---
phase: 84
status: passed
---

# Phase 84 Verification

## Result

Passed.

## Checks

| Requirement | Evidence | Status |
|-------------|----------|--------|
| OUT-01 probability matrix | `compute_speaker_probabilities` emits `188 x 4` probabilities from `188 x 192` hidden frames using maintained `mods.sh2s.*` tensors. | passed |
| OUT-01 frame semantics | Output constants expose `k_frame_shift_seconds = 0.08f`. | passed |
| OUT-02 bounded segments | `decode_segments` writes caller-owned `segment_record` entries and fails when capacity is insufficient. | passed |
| OUT-02 stable labels | `speaker_label` maps `0..3` to `speaker_0` through `speaker_3`. | passed |
| OUT-02 overlap behavior | Segment decoding scans speakers independently, allowing overlapping active regions. | passed |
| OUT-03 deterministic records | Tests cover repeated probability computation, monotonic frame/timestamp fields, and snapshot-friendly fixed records. | passed |

## Verification Commands

- `git diff --check -- CMakeLists.txt src/emel/diarization/sortformer/output/detail.hpp src/emel/diarization/sortformer/output/detail.cpp tests/diarization/sortformer/output/lifecycle_tests.cpp .planning/phases/84-speaker-activity-and-segment-output-contract`
- `cmake --build build/coverage --target emel_tests_bin -j 8`
- `ctest --test-dir build/coverage --output-on-failure -R '^emel_tests_diarization$' -j 1`
- `scripts/quality_gates.sh`

## Results

- Build: passed.
- Focused diarization shard after final label update: passed in `29.42s`.
- Final full quality gate: passed with exit `0`.
- Changed-source coverage from the final full gate:
  - lines: `96.3%`
  - branches: `68.2%`

## Non-Phase Noise

- The final quality gate printed a benchmark regression for `kernel/aarch64/op_soft_max`, then
  printed `warning: benchmark snapshot regression ignored by quality gates`.
