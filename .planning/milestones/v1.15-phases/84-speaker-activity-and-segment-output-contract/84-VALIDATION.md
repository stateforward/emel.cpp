---
phase: 84
slug: speaker-activity-and-segment-output-contract
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-23
---

# Phase 84 - Validation Strategy

## Completion Preconditions

- [x] `84-01-SUMMARY.md` exists
- [x] `84-VERIFICATION.md` exists
- [x] Validation records executable commands and rule-review evidence
- [x] No unresolved manual-only blockers remain in validation scope

## Rule Compliance Review

| Rule Input | Read | Result |
|------------|------|--------|
| `AGENTS.md` | ✅ | The retained proof keeps probability and segment decoding in the owning Sortformer output component and does not add tool-only output logic. |
| `docs/rules/sml.rules.md` | ✅ | Validation scope records existing bounded output helpers and tests only; no mailbox/defer behavior was introduced. |
| `docs/rules/cpp.rules.md` | ✅ | No C++ source changed during this validation backfill. |

*No rule violations found within validation scope.*

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | phase-artifact verification via `rg` |
| **Config file** | none |
| **Quick run command** | `rg -n '^status: passed$|OUT-01|OUT-02|OUT-03' .planning/phases/84-speaker-activity-and-segment-output-contract/{84-VERIFICATION.md,84-01-SUMMARY.md}` |
| **Full suite command** | `rg -n '^requirements-completed:|OUT-01|OUT-02|OUT-03|speaker-activity probabilities and bounded segment records' .planning/phases/84-speaker-activity-and-segment-output-contract/{84-01-SUMMARY.md,84-VERIFICATION.md,84-VALIDATION.md}` |
| **Estimated runtime** | <1 second |

## Manual-Only Verifications

All phase behaviors have automated verification in the cited commands and retained verification
artifact. No unresolved manual-only blockers remain.

## Validation Sign-Off

- [x] Completion preconditions satisfied
- [x] Rule-compliance review recorded
- [x] Executable verification commands documented
- [x] No manual-only blockers remain
- [x] `nyquist_compliant: true` is supported by artifact evidence

**Approval:** approved 2026-04-23
