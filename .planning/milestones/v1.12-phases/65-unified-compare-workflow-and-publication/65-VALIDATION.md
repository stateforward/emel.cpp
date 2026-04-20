---
phase: 65
slug: unified-compare-workflow-and-publication
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-17
---

# Phase 65 — Validation Strategy

## Completion Preconditions

- [x] At least one phase `SUMMARY.md` exists
- [x] Phase `VERIFICATION.md` exists
- [x] ROADMAP / STATE mark the phase complete or ready for validation
- [x] `nyquist_compliant: true` is never set from frontmatter alone

## Rule Compliance Review

| Rule Input | Read | Result |
|------------|------|--------|
| `AGENTS.md` | ✅ | The unified compare workflow remains in maintained tooling, and closeout truth now includes the Phase 66 repair for multi-record C++ publication. |
| `docs/rules/sml.rules.md` | ✅ | No SML actor-orchestration paths are changed within this validation scope. |
| `docs/rules/cpp.rules.md` | ✅ | Bench-tool workflow publication and test coverage remain consistent with current project rules; no blocking violations were found. |

*No rule violations found within validation scope.*

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | doctest via CTest plus maintained workflow commands |
| **Config file** | `tools/bench/CMakeLists.txt` |
| **Quick run command** | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R '^embedding_compare_tests$'` |
| **Full suite command** | `scripts/quality_gates.sh` |
| **Estimated runtime** | ~1500 seconds |

## Sampling Rate

- **After every task commit:** Run `ctest --test-dir build/bench_tools_ninja --output-on-failure -R '^embedding_compare_tests$'`
- **After every plan wave:** Run the maintained compare workflows for Python and C++
- **Before `$gsd-verify-work`:** Workflow artifacts and the full repo gate must both be green
- **Max feedback latency:** 1500 seconds

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 65-01-01 | 01 | 1 | `CMP-01` | — | Operators can run one consistent compare workflow across backend languages | integration | `scripts/bench_embedding_compare.sh --reference-backend te_python_goldens --case-filter text --skip-emel-build --output-dir build/embedding_compare/wrapper_python_text` | ✅ | ✅ green |
| 65-01-02 | 01 | 1 | `CMP-02` | — | Published artifacts record backend, fixture, and reproducibility metadata truthfully | integration | `python3 tools/bench/embedding_compare.py --reference-backend te_python_goldens --emel-runner build/bench_tools_ninja/embedding_generator_bench_runner --output-dir build/embedding_compare/te_python_goldens_all` | ✅ | ✅ green |
| 65-01-03 | 01 | 1 | `CMP-02` | — | Maintained C++ baseline publication remains truthful after the Phase 66 repair | integration | `python3 tools/bench/embedding_compare.py --reference-backend liquid_cpp --emel-runner build/bench_tools_ninja/embedding_generator_bench_runner --output-dir build/embedding_compare/liquid_cpp_compare_text_repaired --case-filter text` | ✅ | ✅ green |
| 65-01-04 | 01 | 1 | `CMP-01` | — | Full repo gate still accepts the maintained unified compare surface | gate | `scripts/quality_gates.sh` | ✅ | ✅ green |

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
- [x] Feedback latency < 1500s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-17
