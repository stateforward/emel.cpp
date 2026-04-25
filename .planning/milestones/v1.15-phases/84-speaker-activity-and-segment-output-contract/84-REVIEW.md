---
phase: 84
status: clean
depth: standard
reviewed: 2026-04-23
---

# Phase 84 Code Review

## Result

Clean.

## Scope

- `CMakeLists.txt`
- `src/emel/diarization/sortformer/output/detail.hpp`
- `src/emel/diarization/sortformer/output/detail.cpp`
- `tests/diarization/sortformer/output/lifecycle_tests.cpp`

## Findings

No open findings.

## Review Notes

- Output helpers use caller-owned probability and segment spans.
- `decode_segments` reports insufficient segment capacity rather than allocating.
- Stable labels are exposed through `speaker_label`.
- Tests cover deterministic probabilities, invalid probability inputs, overlapping segments,
  capacity rejection, threshold rejection, and invalid label indexes.

## Verification

- `cmake --build build/coverage --target emel_tests_bin -j 8`
- `ctest --test-dir build/coverage --output-on-failure -R '^emel_tests_diarization$' -j 1`
- `scripts/quality_gates.sh`
