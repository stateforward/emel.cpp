---
phase: 75-comparability-verdict-and-single-lane-publication-repair
reviewed: 2026-04-21T05:09:31Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - tools/bench/generation_compare.py
  - tools/bench/generation_compare_tests.cpp
  - tools/bench/generation_bench.cpp
  - tools/bench/generation_workloads/lfm2_single_user_hello_max_tokens_1_single_lane.json
  - tools/bench/generation_workloads/README.md
  - docs/benchmarking.md
findings:
  critical: 0
  warning: 2
  info: 1
  total: 3
status: issues_found
---

# Phase 75: Code Review Report

**Reviewed:** 2026-04-21T05:09:31Z
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Reviewed the Phase 75 generation compare wrapper, benchmark wiring, single-lane workload manifest,
tests, and benchmarking docs. The intended single-lane EMEL-only path is present, and comparable
metadata mismatches are checked before output drift comparison. I found two correctness issues in
the drift summarizer plus one missing-test gap around the intended metadata/single-lane contract.

## Warnings

### WR-01: Empty Matching Outputs Are Reported As Drift

**File:** `tools/bench/generation_compare.py:355`

**Issue:** `exact_output_match` is gated by `bool(emel_text)`, and `exact_checksum_match` requires a
non-zero checksum. If both comparable lanes legitimately produce empty output text (for example, an
immediate stop/control token), both records can have `output_bytes=0`, `output_checksum=0`, and no
dumped `output_path`; the summary reports `bounded_drift/output_mismatch` even though the outputs
match. I reproduced this with synthetic EMEL/reference records: the script printed
`status=bounded_drift reason=output_mismatch` with both `output_bytes_delta=0` and
`output_tokens_delta=0`.

**Fix:**
```python
emel_output_bytes = int(emel_record.get("output_bytes", 0))
reference_output_bytes = int(reference_record.get("output_bytes", 0))
both_empty_outputs = emel_output_bytes == 0 and reference_output_bytes == 0
exact_output_match = (both_empty_outputs or bool(emel_text) or bool(reference_text)) and (
  emel_text == reference_text
)
```

Add a unit test with comparable EMEL/reference records that both have `output_bytes=0`,
`output_checksum=0`, and empty `output_path`, and assert `comparison_status=exact_match`.

### WR-02: Shared Prefix Bytes Counts Unicode Code Points, Not Bytes

**File:** `tools/bench/generation_compare.py:263`

**Issue:** `compare_prefix_bytes` compares Python `str` characters and returns the character count,
but the field is published as `shared_prefix_bytes`. Non-ASCII UTF-8 output will undercount the
shared byte prefix, making bounded-drift summaries inconsistent with `output_bytes`.

**Fix:**
```python
def compare_prefix_bytes(lhs: str, rhs: str) -> int:
  lhs_bytes = lhs.encode("utf-8")
  rhs_bytes = rhs.encode("utf-8")
  prefix = 0
  limit = min(len(lhs_bytes), len(rhs_bytes))
  while prefix < limit and lhs_bytes[prefix] == rhs_bytes[prefix]:
    prefix += 1
  return prefix
```

Add a bounded-drift unit test with a shared non-ASCII prefix and assert the byte count, not the
Unicode code-point count.

## Info

### IN-01: Metadata-Mismatch And Empty-Reference Tests Are Partial

**File:** `tools/bench/generation_compare_tests.cpp:416`

**Issue:** The intended contract says comparable summaries reject mismatched
workload/fixture/prompt/formatter/sampling/stop/seed/max-output metadata before output drift.
Current tests cover `sampling_id`, `formatter_contract`, and `max_output_tokens`, but do not cover
`workload_id`, `workload_manifest_path`, fixture identity, prompt identity, `formatter_mode`,
`stop_id`, or `seed`. The single-lane end-to-end test also checks that `raw/reference.jsonl` lacks
the reference backend id, but not that the file is actually empty as documented.

**Fix:** Parameterize the metadata mismatch test over every field in `SUMMARY_METADATA_FIELDS` and
assert the expected reason for each. In the single-lane end-to-end test, assert
`reference_jsonl.empty()` after reading `raw/reference.jsonl`.

---

_Reviewed: 2026-04-21T05:09:31Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
