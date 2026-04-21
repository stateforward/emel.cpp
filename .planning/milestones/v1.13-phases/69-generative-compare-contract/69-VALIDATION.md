---
phase: 69
slug: generative-compare-contract
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-21
---

# Phase 69 — Validation Strategy

## Completion Preconditions

- [x] At least one phase `SUMMARY.md` exists
- [x] Phase `VERIFICATION.md` exists
- [x] ROADMAP / STATE mark the phase complete or ready for validation
- [x] `nyquist_compliant: true` is never set from frontmatter alone

## Rule Compliance Review

| Rule Input | Read | Result |
|------------|------|--------|
| `AGENTS.md` | ✅ | Generation compare contract work stayed in `tools/bench` and kept EMEL/reference lane identity explicit. |
| `docs/rules/sml.rules.md` | ✅ | No `src/` SML actor orchestration changed in this validation scope. |
| `docs/rules/cpp.rules.md` | ✅ | Bench-tool schema and doctest evidence are aligned with existing C++ test conventions. |

*No rule violations found within validation scope.*

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | doctest / CTest / quality gates |
| **Config file** | `tools/bench/CMakeLists.txt` |
| **Quick run command** | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R bench_runner_tests` |
| **Full suite command** | `./scripts/quality_gates.sh` |
| **Estimated runtime** | ~10 minutes |

## Sampling Rate

- **After every task commit:** Run the focused bench runner doctests
- **After every plan wave:** Run `ctest -R bench_runner_tests`
- **Before `$gsd-verify-work`:** Full quality gate must be green
- **Max feedback latency:** 10 minutes

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 69-01-01 | 01 | 1 | `GEN-01` | — | Backend identity remains tooling metadata, not EMEL runtime selection | doctest | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R bench_runner_tests` | ✅ | ✅ green |
| 69-01-02 | 01 | 1 | `GEN-02` | — | EMEL/reference lanes emit the same canonical schema | doctest | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R bench_runner_tests` | ✅ | ✅ green |
| 69-01-03 | 01 | 1 | `ISO-01` | — | Reference runtime objects stay out of `src/` and lane records are explicit | quality | `./scripts/quality_gates.sh` | ✅ | ✅ green |

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
