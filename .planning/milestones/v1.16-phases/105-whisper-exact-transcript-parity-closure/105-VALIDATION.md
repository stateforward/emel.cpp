---
phase: 105
slug: whisper-exact-transcript-parity-closure
status: complete
nyquist_compliant: true
wave_0_complete: true
created: 2026-04-27
---

# Phase 105 - Validation Strategy

Per-phase validation contract for feedback sampling during execution.

---

## Completion Preconditions

- [x] At least one phase SUMMARY.md exists
- [x] Phase VERIFICATION.md exists
- [x] ROADMAP / STATE represent Phase 105 truthfully as transition or supersession work
- [x] nyquist_compliant: true is never set from frontmatter alone

---

## Rule Compliance Review

| Rule Input | Read | Result |
|------------|------|--------|
| `AGENTS.md` | yes | Must not claim direct pinned-artifact parity when EMEL and reference consume different artifacts; maintained-path claims require source-backed live code tracing. |
| `docs/rules/sml.rules.md` | yes | Phase 105 should not introduce SML action/guard changes; any discovered dispatch allocation remains Phase 107 scope. |
| `docs/rules/cpp.rules.md` | yes | Verification commands must avoid treating build artifacts as source truth without provenance checks. |

No rule violations are permitted within Phase 105 transition artifacts.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest through CTest, plus repository bench wrappers |
| Config file | `CMakeLists.txt`, `scripts/bench_whisper_compare.sh`, `scripts/quality_gates.sh` |
| Quick run command | `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` |
| Direct pinned check | `EMEL_WHISPER_EMEL_MODEL="$PWD/build/whisper_reference/whisper-tiny-q8_0-whispercpp.gguf" scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` |
| Full suite command | `EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare scripts/quality_gates.sh` |
| Estimated runtime | bench compare: seconds to minutes depending on existing build; full gate: longer |

---

## Sampling Rate

- After every planning-artifact task: run `rg -n "PARITY-01|CLOSE-01|superseded|normalized" .planning/phases/105-whisper-exact-transcript-parity-closure .planning/ROADMAP.md .planning/STATE.md .planning/REQUIREMENTS.md`
- After evidence refresh: run `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build`
- Before marking Phase 105 complete: run the direct pinned check and record whether it still fails
- Before milestone closeout: full relevant gates remain Phase 108 scope unless requirements are remapped
- Max feedback latency: one phase task

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 105-01-01 | 01 | 1 | Phase 105 transition | T-105-01 | Phase 105 does not falsely claim PARITY-01/CLOSE-01 completion | artifact | `rg -n "requirements-completed|superseded|PARITY-01|CLOSE-01" .planning/phases/105-whisper-exact-transcript-parity-closure` | yes | green |
| 105-01-02 | 01 | 1 | Phase 105 transition | T-105-02 | Evidence records bridge exact-match and direct pinned failure separately | bench/artifact | `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` plus direct pinned check | yes | green |
| 105-01-03 | 01 | 1 | Phase 105 transition | T-105-03 | ROADMAP/STATE hand off final parity to Phases 107/108 without stale closeout claims | artifact | `rg -n "Phase 105|Phase 107|Phase 108|exact_match|normalized" .planning/ROADMAP.md .planning/STATE.md` | yes | green |

Status: pending, green, red, flaky.

---

## Wave 0 Requirements

- [x] Existing bench compare script is available at scripts/bench_whisper_compare.sh
- [x] Existing Phase 105 research exists at 105-RESEARCH.md
- [x] Existing Phase 105 plan records concrete artifact and evidence tasks

Phase 105 validates transition truth only and does not validate PARITY-01 or CLOSE-01 completion.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Approving normalized-GGUF bridge as a narrowed contract | PARITY-01 / CLOSE-01 | Requires explicit user approval under repo reference policy | Do not assume approval; record bridge as unapproved unless the user explicitly accepts it. |

---

## Validation Sign-Off

- [x] Completion preconditions satisfied
- [x] Rule-compliance review recorded
- [x] All tasks have automated checks or documented manual-only approval
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all missing references
- [x] No watch-mode flags
- [x] Feedback latency remains within one phase task
- [x] nyquist_compliant: true set in frontmatter only after evidence passes

**Approval:** approved 2026-04-27
