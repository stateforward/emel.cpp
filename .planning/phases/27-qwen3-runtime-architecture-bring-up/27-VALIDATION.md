---
phase: 27
slug: qwen3-runtime-architecture-bring-up
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-27
---

# Phase 27 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | doctest via CMake/CTest |
| **Config file** | `CMakeLists.txt` + `tools/paritychecker/CMakeLists.txt` |
| **Quick run command** | `./build/zig/emel_tests_bin --test-case='*qwen3*execution*' --no-breaks` |
| **Full suite command** | `scripts/quality_gates.sh` |
| **Estimated runtime** | ~30 seconds for quick feedback |

---

## Sampling Rate

- **After every task commit:** Run that task's `<automated>` command
- **After every plan wave:** Run `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds for the quick-feedback lane

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 27-01-01 | 01 | 1 | RUN-02 | unit | `./build/zig/emel_tests_bin --test-case='*qwen3*execution*' --no-breaks` | ✅ | ✅ green |
| 27-01-02 | 01 | 1 | RUN-02 | unit | `./build/zig/emel_tests_bin --test-case='*qwen3*execution*,*qwen3*quantized*' --no-breaks` | ✅ | ✅ green |
| 27-02-01 | 02 | 2 | RUN-01 | unit + subprocess | `./build/zig/emel_tests_bin --test-case='*qwen3*generator*' --no-breaks && ./build/paritychecker_zig/paritychecker_tests --test-case='*qwen3*generation*' --no-breaks` | ✅ | ✅ green |
| 27-02-02 | 02 | 2 | RUN-01, RUN-02 | repo gate | `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests && scripts/quality_gates.sh` | ✅ | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- Existing infrastructure covers all phase requirements.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Maintained Qwen runtime output remains narrow to the canonical fixture and does not imply broader Qwen-family coverage | RUN-01, RUN-02 | Automated tests can prove canonical behavior, but human review is still needed to confirm operator-facing wording stays narrow and explicit | Run maintained paritychecker generation on `tests/models/Qwen3-0.6B-Q8_0.gguf` in temp-baseline write mode, inspect output/help text, and confirm it names only the canonical slice plus explicit formatter/runtime contracts and does not advertise broader Qwen-family or stored parity completion |

Manual review completed on 2026-03-28: maintained generation output stays anchored to the canonical
fixture and explicit formatter/runtime contracts, with no broader Qwen-family claims.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency <= 30s in the quick-feedback lane
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validation complete
