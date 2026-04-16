---
phase: {N}
slug: {phase-slug}
status: draft
nyquist_compliant: false
wave_0_complete: false
created: {date}
---

# Phase {N} — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Completion Preconditions

- [ ] At least one phase `SUMMARY.md` exists
- [ ] Phase `VERIFICATION.md` exists
- [ ] ROADMAP / STATE mark the phase complete or ready for validation
- [ ] `nyquist_compliant: true` is never set from frontmatter alone

---

## Rule Compliance Review

| Rule Input | Read | Result |
|------------|------|--------|
| `AGENTS.md` | ✅ / ❌ | {relevant constraints or blocking gap} |
| `docs/rules/sml.rules.md` | ✅ / ❌ | {relevant constraints or blocking gap} |
| `docs/rules/cpp.rules.md` | ✅ / ❌ | {relevant constraints or blocking gap} |

*If no blocking rule issues were found: "No rule violations found within validation scope."*

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | {pytest 7.x / jest 29.x / vitest / go test / other} |
| **Config file** | {path or "none — Wave 0 installs"} |
| **Quick run command** | `{quick command}` |
| **Full suite command** | `{full command}` |
| **Estimated runtime** | ~{N} seconds |

---

## Sampling Rate

- **After every task commit:** Run `{quick run command}`
- **After every plan wave:** Run `{full suite command}`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** {N} seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| {N}-01-01 | 01 | 1 | REQ-{XX} | T-{N}-01 / — | {expected secure behavior or "N/A"} | unit | `{command}` | ✅ / ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `{tests/test_file.py}` — stubs for REQ-{XX}
- [ ] `{tests/conftest.py}` — shared fixtures
- [ ] `{framework install}` — if no framework detected

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| {behavior} | REQ-{XX} | {reason} | {steps} |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] Completion preconditions satisfied
- [ ] Rule-compliance review recorded
- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < {N}s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** {pending / approved YYYY-MM-DD}
