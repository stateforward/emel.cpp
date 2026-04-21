---
phase: 71
slug: maintained-reference-backend-integration
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-21
---

# Phase 71 — Validation Strategy

## Completion Preconditions

- [x] At least one phase `SUMMARY.md` exists
- [x] Phase `VERIFICATION.md` exists
- [x] ROADMAP / STATE mark the phase complete or ready for validation
- [x] `nyquist_compliant: true` is never set from frontmatter alone

## Rule Compliance Review

| Rule Input | Read | Result |
|------------|------|--------|
| `AGENTS.md` | ✅ | Reference backend setup stays in manifest/wrapper tooling and does not mutate EMEL runtime ownership. |
| `docs/rules/sml.rules.md` | ✅ | No `src/` SML actor orchestration changed in this validation scope. |
| `docs/rules/cpp.rules.md` | ✅ | Failure-path coverage is handled by dedicated doctest cases. |

*No rule violations found within validation scope.*

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | doctest / CTest |
| **Config file** | `tools/bench/CMakeLists.txt` |
| **Quick run command** | `./build/bench_tools_ninja/generation_compare_tests` |
| **Full suite command** | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R generation_compare_tests` |
| **Estimated runtime** | ~30 seconds |

## Sampling Rate

- **After every task commit:** Run `generation_compare_tests`
- **After every plan wave:** Run CTest for `generation_compare_tests`
- **Before `$gsd-verify-work`:** CTest must be green
- **Max feedback latency:** 30 seconds

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 71-01-01 | 01 | 1 | `REF-01` | — | Maintained reference backend is selected through manifest tooling | integration | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R generation_compare_tests` | ✅ | ✅ green |
| 71-01-02 | 01 | 1 | `REF-02` | — | Backend setup remains reference-lane tooling | integration | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R generation_compare_tests` | ✅ | ✅ green |
| 71-01-03 | 01 | 1 | `REF-03` | — | Backend errors become explicit reference-lane error records | doctest | `./build/bench_tools_ninja/generation_compare_tests` | ✅ | ✅ green |

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
- [x] Feedback latency < 30 seconds
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-21
