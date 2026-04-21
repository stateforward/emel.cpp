---
phase: 75-comparability-verdict-and-single-lane-publication-repair
plan: 01
status: complete
completed: 2026-04-21T05:30:08Z
requirements-completed:
  - WRK-03
  - CMP-03
  - PRF-01
---

# Phase 75 Summary

## Changes

- Added `lfm2_single_user_hello_max_tokens_1_single_lane_v1` as a maintained local
  single-lane generation workload manifest and registered it in `generation_bench.cpp`.
- Updated `generation_compare.py` so selected non-parity/non-comparable workloads run EMEL only,
  leave `raw/reference.jsonl` empty, and publish `non_comparable` with reason
  `single_lane_emel_workload`.
- Expanded summary metadata and verdict checks to include fixture identity, prompt fixture,
  formatter contract, sampling id, stop id, seed, and max token budget before output comparison.
- Added regression tests for ignored metadata mismatches and for a real wrapper-level single-lane
  workflow.
- Fixed review findings so empty comparable outputs are exact matches and shared-prefix metrics
  count UTF-8 bytes rather than Unicode code points.
- Updated benchmark documentation and workload manifest docs to describe single-lane publication
  behavior and the expanded comparability contract.

## Evidence

- `./build/bench_tools_ninja/generation_compare_tests`
- `ctest --test-dir build/bench_tools_ninja --output-on-failure -R generation_compare_tests`
- `./scripts/quality_gates.sh`
- `75-REVIEW-FIX.md`

## Notes

- The LFM2 single-lane workload is a maintained operator workflow proof for non-comparable
  publication. It is not a parity claim.
- The full quality gate passed with the existing warning-tolerated benchmark snapshot regression
  for `kernel/aarch64/op_soft_max`.
