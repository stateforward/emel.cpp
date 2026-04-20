---
phase: 66
slug: repair-unified-compare-publication
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-17
---

# Phase 66 — Validation Strategy

## Completion Preconditions

- [x] At least one phase `SUMMARY.md` exists
- [x] Phase `VERIFICATION.md` exists
- [x] ROADMAP / STATE mark the phase complete or ready for validation
- [x] `nyquist_compliant: true` is never set from frontmatter alone

## Rule Compliance Review

| Rule Input | Read | Result |
|------------|------|--------|
| `AGENTS.md` | ✅ | Reproduced the reported bug with a failing focused doctest before fixing the compare driver; the repair remains confined to tooling and verification surfaces. |
| `docs/rules/sml.rules.md` | ✅ | No SML actor-orchestration paths are touched by this tooling-only repair. |
| `docs/rules/cpp.rules.md` | ✅ | The change stays within existing bench-tool and test patterns; no blocking C++ rule conflicts were found within validation scope. |

*No rule violations found within validation scope.*

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | doctest via CTest |
| **Config file** | `tools/bench/CMakeLists.txt` |
| **Quick run command** | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R '^embedding_compare_tests$'` |
| **Full suite command** | `python3 tools/bench/embedding_compare.py --reference-backend liquid_cpp --emel-runner build/bench_tools_ninja/embedding_generator_bench_runner --output-dir build/embedding_compare/liquid_cpp_compare_text_repaired --case-filter text` |
| **Estimated runtime** | ~60 seconds |

## Sampling Rate

- **After every task commit:** Run `ctest --test-dir build/bench_tools_ninja --output-on-failure -R '^embedding_compare_tests$'`
- **After every plan wave:** Run the repaired maintained `liquid_cpp` compare command
- **Before `$gsd-verify-work`:** The focused compare test and maintained compare publication must both be green
- **Max feedback latency:** 60 seconds

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 66-01-01 | 01 | 1 | `CPP-01` | — | Regression reproduces the lossy publication bug before the fix | unit | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R '^embedding_compare_tests$'` | ✅ | ✅ green |
| 66-01-02 | 01 | 1 | `CMP-01` | — | Compare summary preserves every emitted maintained reference result | integration | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R '^embedding_compare_tests$'` | ✅ | ✅ green |
| 66-01-03 | 01 | 1 | `CMP-02` | — | Published artifacts include both maintained baseline backend identities | integration | `python3 tools/bench/embedding_compare.py --reference-backend liquid_cpp --emel-runner build/bench_tools_ninja/embedding_generator_bench_runner --output-dir build/embedding_compare/liquid_cpp_compare_text_repaired --case-filter text` | ✅ | ✅ green |

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
- [x] Feedback latency < 60s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-17
