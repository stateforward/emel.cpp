---
phase: 237
slug: direct-tensor-staged-offset-contract-repair
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-08
---

# Phase 237 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest via `emel_tests_bin`; CTest; scoped quality gate |
| Config file | `CMakeLists.txt`, `scripts/quality_gates.sh` |
| Quick run command | `./build/emel_tests_bin --test-case="model_tensor_request_staged_load_*"` |
| Gate command | `EMEL_QUALITY_GATES_CHANGED_FILES="src/emel/model/tensor/actions.hpp src/emel/model/tensor/guards.hpp tests/model/tensor/lifecycle_tests.cpp .planning/phases/237-direct-tensor-staged-offset-contract-repair/237-CONTEXT.md .planning/phases/237-direct-tensor-staged-offset-contract-repair/237-01-PLAN.md" scripts/quality_gates.sh` |
| Estimated runtime | ~3 minutes for the recorded scoped gate |

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 237-01-01 | 01 | 1 | TNX-01, TST-01 | direct staged offset ignored | Public tensor staged-load dispatch copies the requested nonzero-offset source window. | doctest | `./build/emel_tests_bin --test-case="model_tensor_request_staged_load_applies_nonzero_file_offset"` | yes | green |
| 237-01-02 | 01 | 1 | TNX-03, TNX-04, TST-02 | silent or ambiguous terminal outcome | Success and failure remain observable through explicit `_done` / `_error` callbacks and tensor state inspection. | doctest subset | `./build/emel_tests_bin --test-case="model_tensor_request_staged_load_*"` | yes | green |
| 237-01-03 | 01 | 1 | TNX-01 | invalid offset source coverage | Tensor guard validates `source_buffer_bytes` covers `file_offset + byte_size` before action pointer arithmetic. | scoped gate + source review | `scripts/quality_gates.sh` with changed-file scope | yes | green |
| 237-01-04 | 01 | 1 | TST-01, TST-02 | stale integration evidence | Maintained model/io test shard and quality gate pass without benchmark override. | CTest + gate | `ctest --test-dir build -R '^emel_tests_model_and_batch$' --output-on-failure` | yes | green |

## Wave 0 Requirements

The added regression doctest covers the previously missing nonzero-offset direct
tensor staged-load path. Existing staged tensor failure tests cover the direct
route's explicit error publication.

## Manual-Only Verifications

No manual-only verifications are required for this phase.

## Validation Sign-Off

- [x] All tasks have automated verification.
- [x] Sampling continuity is preserved by doctest subset, CTest shard, and scoped quality gate.
- [x] Wave 0 covers the audit gap.
- [x] No watch-mode flags.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-08
