---
phase: 62
slug: reference-backend-contract
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-17
---

# Phase 62 — Validation Strategy

## Completion Preconditions

- [x] At least one phase `SUMMARY.md` exists
- [x] Phase `VERIFICATION.md` exists
- [x] ROADMAP / STATE mark the phase complete or ready for validation
- [x] `nyquist_compliant: true` is never set from frontmatter alone

## Rule Compliance Review

| Rule Input | Read | Result |
|------------|------|--------|
| `AGENTS.md` | ✅ | The compare contract remains in tooling and preserves explicit EMEL/reference isolation; no blocking closeout drift was found within validation scope. |
| `docs/rules/sml.rules.md` | ✅ | No SML actor-orchestration code changed in this validation scope. |
| `docs/rules/cpp.rules.md` | ✅ | Bench-tool schema wiring and validation commands remain consistent with current C++ project rules; no blocking violations were found. |

*No rule violations found within validation scope.*

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | bench-tool command verification |
| **Config file** | `tools/bench/CMakeLists.txt` |
| **Quick run command** | `cmake --build build/bench_tools_ninja --parallel --target embedding_generator_bench_runner` |
| **Full suite command** | `python3 tools/bench/embedding_compare.py --reference-backend te_python_goldens --emel-runner build/bench_tools_ninja/embedding_generator_bench_runner --output-dir build/embedding_compare/te_python_goldens_all` |
| **Estimated runtime** | ~60 seconds |

## Sampling Rate

- **After every task commit:** Rebuild `embedding_generator_bench_runner`
- **After every plan wave:** Run the maintained Python compare command
- **Before `$gsd-verify-work`:** Both lanes must publish the same canonical schema
- **Max feedback latency:** 60 seconds

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 62-01-01 | 01 | 1 | `REF-01` | — | Backend selection stays in tooling and does not alter the EMEL runner path | build | `cmake --build build/bench_tools_ninja --parallel --target embedding_generator_bench_runner` | ✅ | ✅ green |
| 62-01-02 | 01 | 1 | `REF-02` | — | EMEL and reference lanes emit the same canonical compare schema | integration | `python3 tools/bench/embedding_compare.py --reference-backend te_python_goldens --emel-runner build/bench_tools_ninja/embedding_generator_bench_runner --output-dir build/embedding_compare/te_python_goldens_all` | ✅ | ✅ green |
| 62-01-03 | 01 | 1 | `ISO-01` | — | Reference runtime state remains isolated from the EMEL lane | integration | `python3 tools/bench/embedding_compare.py --reference-backend te_python_goldens --emel-runner build/bench_tools_ninja/embedding_generator_bench_runner --output-dir build/embedding_compare/te_python_goldens_all` | ✅ | ✅ green |

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
