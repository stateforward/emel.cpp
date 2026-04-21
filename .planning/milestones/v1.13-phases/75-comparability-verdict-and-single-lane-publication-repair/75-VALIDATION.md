---
phase: 75
slug: comparability-verdict-and-single-lane-publication-repair
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-21
---

# Phase 75 — Validation Strategy

## Completion Preconditions

- [x] At least one phase `SUMMARY.md` exists
- [x] Phase `VERIFICATION.md` exists
- [x] ROADMAP / STATE mark the phase complete or ready for validation
- [x] `nyquist_compliant: true` is never set from frontmatter alone

## Rule Compliance Review

| Rule Input | Read | Result |
|------------|------|--------|
| `AGENTS.md` | ✅ | Single-lane publication proof stays in benchmark tooling and manifest docs; no runtime fallback was added. |
| `docs/rules/sml.rules.md` | ✅ | No `src/` SML actor orchestration changed in this validation scope. |
| `docs/rules/cpp.rules.md` | ✅ | Review fixes are covered by focused doctests and quality gates. |

*No rule violations found within validation scope.*

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | doctest / CTest / quality gates |
| **Config file** | `tools/bench/CMakeLists.txt` |
| **Quick run command** | `./build/bench_tools_ninja/generation_compare_tests` |
| **Full suite command** | `./scripts/quality_gates.sh` |
| **Estimated runtime** | ~10 minutes |

## Sampling Rate

- **After every task commit:** Run `generation_compare_tests`
- **After every plan wave:** Run CTest for `generation_compare_tests`
- **Before `$gsd-verify-work`:** Full quality gate must be green
- **Max feedback latency:** 10 minutes

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 75-01-01 | 01 | 1 | `WRK-03` | — | Material metadata mismatches publish `non_comparable` before drift checks | doctest | `./build/bench_tools_ninja/generation_compare_tests` | ✅ | ✅ green |
| 75-01-02 | 01 | 1 | `CMP-03` | — | Selected single-lane workloads publish `non_comparable` without reference errors | integration | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R generation_compare_tests` | ✅ | ✅ green |
| 75-01-03 | 01 | 1 | `PRF-01` | — | Maintained multi-engine and single-lane workflows run end to end | integration | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R generation_compare_tests` | ✅ | ✅ green |

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

## Manual-Only Verifications

All phase behaviors have automated verification.

## Validation Sign-Off

- [x] Completion preconditions satisfied
- [x] Rule-compliance review recorded
- [x] All tasks have automated verify coverage
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency < 10 minutes
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-21
