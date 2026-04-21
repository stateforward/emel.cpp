---
phase: 70
slug: reproducible-generation-workload-contract
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-21
---

# Phase 70 — Validation Strategy

## Completion Preconditions

- [x] At least one phase `SUMMARY.md` exists
- [x] Phase `VERIFICATION.md` exists
- [x] ROADMAP / STATE mark the phase complete or ready for validation
- [x] `nyquist_compliant: true` is never set from frontmatter alone

## Rule Compliance Review

| Rule Input | Read | Result |
|------------|------|--------|
| `AGENTS.md` | ✅ | Workload truth is checked-in manifest data and benchmark tooling; no public runtime scope widened. |
| `docs/rules/sml.rules.md` | ✅ | No SML actor orchestration changed in this validation scope. |
| `docs/rules/cpp.rules.md` | ✅ | Doctest coverage validates manifest-driven output through existing bench test targets. |

*No rule violations found within validation scope.*

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | doctest / CTest / quality gates |
| **Config file** | `tools/bench/CMakeLists.txt` |
| **Quick run command** | `./build/bench_tools_ninja/bench_runner_tests --test-case="bench_runner generation jsonl emits manifest-driven workload metadata and explicit comparability"` |
| **Full suite command** | `./scripts/quality_gates.sh` |
| **Estimated runtime** | ~10 minutes |

## Sampling Rate

- **After every task commit:** Run the manifest-driven JSONL doctest
- **After every plan wave:** Run `ctest -R bench_runner_tests`
- **Before `$gsd-verify-work`:** Full quality gate must be green
- **Max feedback latency:** 10 minutes

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 70-01-01 | 01 | 1 | `WRK-01` | — | Workloads are selected from explicit checked-in manifests | doctest | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R bench_runner_tests` | ✅ | ✅ green |
| 70-01-02 | 01 | 1 | `WRK-02` | — | JSONL records preserve replay and mismatch metadata | doctest | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R bench_runner_tests` | ✅ | ✅ green |
| 70-01-03 | 01 | 1 | `WRK-03` | — | Single-lane workloads are marked non-comparable instead of parity | doctest | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R bench_runner_tests` | ✅ | ✅ green |

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
