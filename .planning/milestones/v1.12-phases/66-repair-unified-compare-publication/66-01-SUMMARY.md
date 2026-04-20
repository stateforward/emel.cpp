---
phase: 66-repair-unified-compare-publication
plan: 01
status: complete
completed: 2026-04-17
requirements-completed:
  - CPP-01
  - CMP-01
  - CMP-02
---

# Phase 66 Summary

## Outcome

Phase 66 is complete. The unified compare publication path now preserves every emitted maintained
reference result for the repaired `liquid_cpp` text workflow instead of silently dropping the
second baseline record.

## Delivered

- Added a focused regression test in `tools/bench/embedding_compare_tests.cpp` that reproduces the
  dropped-record bug when multiple reference results share one `compare_group`.
- Updated `tools/bench/embedding_compare.py` so summary generation emits one published group entry
  per preserved record pairing rather than selecting only the first matching reference record.
- Republished maintained `liquid_cpp` compare evidence under
  `build/embedding_compare/liquid_cpp_compare_text_repaired/`.

## Published Repair Evidence

- `embedding_compare_tests` now covers duplicate reference groups and passes.
- The repaired maintained `liquid_cpp` compare run now publishes both baseline text backends:
  - `cpp.reference.arctic_s`
  - `cpp.reference.embeddinggemma_300m`
- Operator-visible stdout now prints two `text/red_square/full_dim` baseline lines instead of one.

## Verification Result

- `ctest --test-dir build/bench_tools_ninja --output-on-failure -R '^embedding_compare_tests$'`
  passed.
- `python3 tools/bench/embedding_compare.py --reference-backend liquid_cpp ... --case-filter text`
  exited `0` and wrote a repaired `compare_summary.json` that includes both maintained baseline
  backends.
