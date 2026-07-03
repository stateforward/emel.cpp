---
phase: 233
slug: public-loader-and-maintained-entrypoints
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-08
---

# Phase 233 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest via `emel_tests_bin`; CTest; source-scan guardrails |
| Config file | `CMakeLists.txt` |
| Quick run command | `ctest --test-dir build --output-on-failure -R 'emel_tests_(io\|model)'` |
| Full suite command | `EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_PARALLEL=0 scripts/quality_gates.sh` |
| Estimated runtime | focused lane under 1 minute; full gate recorded at closeout |

## Sampling Rate

- After loader/tool entrypoint edits: run model and I/O focused CTest lanes.
- Before milestone sign-off: run full quality gate to cover maintained benches, parity, fuzz, lint,
  and coverage.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 233-01-01 | 01 | 1 | PUB-01 | loader public contract | `model/loader` selects/reports staged use through public loader contracts. | doctest + source scan | `ctest --test-dir build --output-on-failure -R 'emel_tests_(io\|model)'` | yes | green |
| 233-01-02 | 01 | 1 | PUB-02 | benchmark public contract | Benchmark entrypoints bind staged read through `bind_model_load_io_strategy`. | source scan + full gate | `EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_PARALLEL=0 scripts/quality_gates.sh` | yes | green |
| 233-01-03 | 01 | 1 | PUB-03 | parity public contract | Paritychecker reports staged-read usage through public model-loader outcomes. | source scan + full gate | `EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_PARALLEL=0 scripts/quality_gates.sh` | yes | green |
| 233-01-04 | 01 | 1 | PUB-04 | probe public contract | Embedded-size probe uses public model-load strategy binding. | source scan + full gate | `EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_PARALLEL=0 scripts/quality_gates.sh` | yes | green |
| 233-01-05 | 01 | 1 | PUB-05 | no duplicate staged loop | Maintained tools do not embed private staged-read detail reach-through or duplicate staging loop. | source scan | `ctest --test-dir build --output-on-failure -R 'emel_tests_(io\|model)'` | yes | green |

## Wave 0 Requirements

Existing model/loader, I/O loader, and source-scan tests cover all phase requirements.

## Manual-Only Verifications

All phase behaviors have automated or source-scan verification.

## Validation Sign-Off

- [x] All tasks have automated verification.
- [x] Sampling continuity is preserved by focused CTest lanes and full gate closeout.
- [x] Wave 0 covers missing references.
- [x] No watch-mode flags.
- [x] `nyquist_compliant: true` set in frontmatter.

**Approval:** approved 2026-05-08
