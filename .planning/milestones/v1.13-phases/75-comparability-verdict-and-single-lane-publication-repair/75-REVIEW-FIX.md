---
phase: 75-comparability-verdict-and-single-lane-publication-repair
review: 75-REVIEW.md
status: fixed
fixed: 2026-04-21T05:30:08Z
---

# Phase 75 Review Fix Summary

## Fixed Findings

### WR-01: Empty Matching Outputs Are Reported As Drift

- Updated `generation_compare.py` so comparable records with zero output bytes on both lanes are
  reported as `exact_match` when their read output text is equal.
- Added `generation compare reports empty comparable outputs as exact matches`.

### WR-02: Shared Prefix Bytes Counts Unicode Code Points, Not Bytes

- Updated `compare_prefix_bytes` to compare UTF-8 encoded bytes.
- Updated shared-prefix fraction denominator to use UTF-8 byte lengths.
- Added `generation compare reports shared prefixes in UTF-8 bytes`.

## Additional Review Cleanup

- Strengthened the maintained single-lane wrapper test to assert that `raw/reference.jsonl` is
  empty, matching the documented selected single-lane behavior.

## Verification

- `python3 -m py_compile tools/bench/generation_compare.py`
- `./build/bench_tools_ninja/generation_compare_tests`
- `ctest --test-dir build/bench_tools_ninja --output-on-failure -R generation_compare_tests`
- `./scripts/quality_gates.sh`
