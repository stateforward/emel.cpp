---
phase: 64
slug: cpp-reference-backend-integration
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-17
---

# Phase 64 — Validation Strategy

## Completion Preconditions

- [x] At least one phase `SUMMARY.md` exists
- [x] Phase `VERIFICATION.md` exists
- [x] ROADMAP / STATE mark the phase complete or ready for validation
- [x] `nyquist_compliant: true` is never set from frontmatter alone

## Rule Compliance Review

| Rule Input | Read | Result |
|------------|------|--------|
| `AGENTS.md` | ✅ | C++ backend setup remains confined to wrapper/tooling paths and does not leak into `src/`; no blocking closeout drift was found. |
| `docs/rules/sml.rules.md` | ✅ | No SML actor-orchestration code is touched within this validation scope. |
| `docs/rules/cpp.rules.md` | ✅ | The maintained compare integration stays in C++ bench tooling and wrapper scripts; no blocking rule conflicts were found. |

*No rule violations found within validation scope.*

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | shell command verification plus compare integration |
| **Config file** | `scripts/bench_embedding_reference_liquid.sh` and `tools/bench/CMakeLists.txt` |
| **Quick run command** | `scripts/bench_embedding_reference_liquid.sh --build-only` |
| **Full suite command** | `python3 tools/bench/embedding_compare.py --reference-backend liquid_cpp --emel-runner build/bench_tools_ninja/embedding_generator_bench_runner --output-dir build/embedding_compare/liquid_cpp_compare_text --case-filter text` |
| **Estimated runtime** | ~90 seconds |

## Sampling Rate

- **After every task commit:** Rebuild the maintained Liquid wrapper with `--build-only`
- **After every plan wave:** Run the maintained C++ wrapper and compare driver
- **Before `$gsd-verify-work`:** The C++ backend must emit canonical JSONL and truthful baseline output
- **Max feedback latency:** 90 seconds

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 64-01-01 | 01 | 1 | `CPP-01` | — | Maintained C++ backend runs through the shared manifest contract | smoke | `EMEL_EMBEDDING_BENCH_FORMAT=jsonl EMEL_EMBEDDING_RESULT_DIR=build/embedding_compare/liquid_cpp_text EMEL_BENCH_CASE_FILTER=arctic_s scripts/bench_embedding_reference_liquid.sh --run-only` | ✅ | ✅ green |
| 64-01-02 | 01 | 1 | `CPP-02` | — | C++ backend setup stays outside the EMEL compute path | integration | `python3 tools/bench/embedding_compare.py --reference-backend liquid_cpp --emel-runner build/bench_tools_ninja/embedding_generator_bench_runner --output-dir build/embedding_compare/liquid_cpp_compare_text --case-filter text` | ✅ | ✅ green |

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
- [x] Feedback latency < 90s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-17
