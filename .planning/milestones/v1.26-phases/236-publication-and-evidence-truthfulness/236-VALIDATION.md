---
phase: 236
slug: publication-and-evidence-truthfulness
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-08
---

# Phase 236 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest via `emel_tests_bin`; CTest; benchmark/parity/fuzz/lint quality gates |
| Config file | `CMakeLists.txt`, `scripts/quality_gates.sh`, `scripts/bench.sh` |
| Quick run command | `ctest --test-dir build --output-on-failure -R lint_snapshot` |
| Full suite command | `EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_PARALLEL=0 scripts/quality_gates.sh` |
| Estimated runtime | ~13 minutes for the recorded serial full gate |

## Sampling Rate

- After doc or snapshot edits: run the relevant maintained lane (`lint_snapshot` or `bench.sh`).
- Before milestone sign-off: run the serial full quality gate.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 236-01-01 | 01 | 1 | DOC-01 | stale publication truth | Maintained docs state staged-read implementation and access path accurately. | source diff review | `rg "staged" README.md docs/roadmap.md docs/templates/README.md.j2` | yes | green |
| 236-01-02 | 01 | 1 | LNT-01 | stale lint snapshot | Lint snapshot is updated only through maintained workflow and passes afterward. | CTest + script | `ctest --test-dir build --output-on-failure -R lint_snapshot` | yes | green |
| 236-01-03 | 01 | 1 | BNH-01 | stale benchmark baseline | Benchmark snapshot changes follow the maintained benchmark update/compare workflow. | benchmark script + full gate | `scripts/bench.sh --snapshot --compare` | yes | green |
| 236-01-04 | 01 | 1 | EVI-01 | misleading staged evidence | Maintained parity/benchmark labels are based on `used_io_strategy`, not requested capability alone. | source scan + full gate | `EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_PARALLEL=0 scripts/quality_gates.sh` | yes | green |

## Wave 0 Requirements

Existing lint, benchmark, parity, fuzz, and full quality-gate infrastructure covers all phase
requirements.

## Manual-Only Verifications

All phase behaviors have automated or source-backed verification.

## Validation Sign-Off

- [x] All tasks have automated verification.
- [x] Sampling continuity is preserved by maintained lane commands and full gate.
- [x] Wave 0 covers missing references.
- [x] No watch-mode flags.
- [x] Feedback latency policy repaired by reducing default benchmark iterations.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-08
