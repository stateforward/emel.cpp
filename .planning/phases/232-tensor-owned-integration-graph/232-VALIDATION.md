---
phase: 232
slug: tensor-owned-integration-graph
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-08
---

# Phase 232 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest via `emel_tests_bin`; CTest; quality gate scripts |
| Config file | `CMakeLists.txt` |
| Quick run command | `ctest --test-dir build --output-on-failure -R 'emel_tests_(io\|model)'` |
| Full suite command | `EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_PARALLEL=0 scripts/quality_gates.sh` |
| Estimated runtime | focused lane under 1 minute; full gate recorded at closeout |

## Sampling Rate

- After tensor/staged integration edits: run the focused I/O and model CTest lanes.
- Before milestone sign-off: rely on the serial full quality gate recorded in Phase 236 for
  repo-level regression closure.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 232-01-01 | 01 | 1 | TNX-01 | public event boundary | `model/tensor` initiates staged loading through public `emel/io` staged-read events. | doctest + source scan | `ctest --test-dir build --output-on-failure -R 'emel_tests_(io\|model)'` | yes | green |
| 232-01-02 | 01 | 1 | TNX-02 | residency ownership | Tensor lifecycle remains owned by `model/tensor`. | doctest + source scan | `ctest --test-dir build --output-on-failure -R 'emel_tests_(io\|model)'` | yes | green |
| 232-01-03 | 01 | 1 | TNX-03 | terminal success visibility | Staged-load success has explicit observable terminal done state/event. | doctest | `ctest --test-dir build --output-on-failure -R 'emel_tests_(io\|model)'` | yes | green |
| 232-01-04 | 01 | 1 | TNX-04 | terminal failure visibility | Staged-load failure has explicit observable terminal error state/event. | doctest | `ctest --test-dir build --output-on-failure -R 'emel_tests_(io\|model)'` | yes | green |

## Wave 0 Requirements

Existing model/tensor and I/O tests cover all phase requirements.

## Manual-Only Verifications

All phase behaviors have automated or source-scan verification.

## Notes

Phase-local scoped gate history recorded unrelated benchmark/parity failures. The milestone
closeout full gate later passed with exit 0, so those historical failures are not open blockers.

## Validation Sign-Off

- [x] All tasks have automated verification.
- [x] Sampling continuity is preserved by focused model/I/O CTest lanes.
- [x] Wave 0 covers missing references.
- [x] No watch-mode flags.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-08
