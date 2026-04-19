---
phase: 63
slug: python-reference-backend
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-17
---

# Phase 63 — Validation Strategy

## Completion Preconditions

- [x] At least one phase `SUMMARY.md` exists
- [x] Phase `VERIFICATION.md` exists
- [x] ROADMAP / STATE mark the phase complete or ready for validation
- [x] `nyquist_compliant: true` is never set from frontmatter alone

## Rule Compliance Review

| Rule Input | Read | Result |
|------------|------|--------|
| `AGENTS.md` | ✅ | Python backend work stays in maintained tooling and reports explicit lane errors instead of masking EMEL results; no blocking closeout drift was found. |
| `docs/rules/sml.rules.md` | ✅ | No SML actor-orchestration paths are modified within this validation scope. |
| `docs/rules/cpp.rules.md` | ✅ | The phase stays in bench-tool and script surfaces with focused test coverage; no blocking C++ rule issues were found. |

*No rule violations found within validation scope.*

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Python command verification plus doctest via CTest |
| **Config file** | `tools/bench/CMakeLists.txt` |
| **Quick run command** | `python3 -m py_compile tools/bench/embedding_compare.py tools/bench/embedding_reference_python.py` |
| **Full suite command** | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R '^embedding_compare_tests$'` |
| **Estimated runtime** | ~30 seconds |

## Sampling Rate

- **After every task commit:** Run `python3 -m py_compile tools/bench/embedding_compare.py tools/bench/embedding_reference_python.py`
- **After every plan wave:** Run the maintained Python golden backend plus focused compare tests
- **Before `$gsd-verify-work`:** The Python backend must emit canonical compare records and tests must stay green
- **Max feedback latency:** 30 seconds

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 63-01-01 | 01 | 1 | `PY-01` | — | Maintained Python backend emits canonical compare records through the shared contract | smoke | `EMEL_EMBEDDING_BENCH_FORMAT=jsonl EMEL_EMBEDDING_RESULT_DIR=build/embedding_compare/python_goldens EMEL_BENCH_CASE_FILTER=text python3 tools/bench/embedding_reference_python.py --backend te75m_goldens` | ✅ | ✅ green |
| 63-01-02 | 01 | 1 | `PY-02` | — | Python lane failures surface explicitly instead of corrupting EMEL results | unit | `ctest --test-dir build/bench_tools_ninja --output-on-failure -R '^embedding_compare_tests$'` | ✅ | ✅ green |

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
- [x] Feedback latency < 30s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-04-17
