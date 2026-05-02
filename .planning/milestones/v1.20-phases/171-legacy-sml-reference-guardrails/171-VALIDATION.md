---
phase: 171
slug: legacy-sml-reference-guardrails
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-01
---

# Phase 171 — Validation Strategy

## Quick Feedback Lane

- `scripts/check_legacy_sml_surface.sh`
- `scripts/lint_snapshot.sh`
- `EMEL_QUALITY_GATES_CHANGED_FILES='scripts/check_legacy_sml_surface.sh scripts/quality_gates.sh tests/sm/sm_policy_tests.cpp CMakeLists.txt AGENTS.md docs/plans/rearchitecture.plan.md .planning/REQUIREMENTS.md' EMEL_QUALITY_GATES_DOCS=always scripts/quality_gates.sh`

## Rule Compliance Evidence

- Phase 176 contains the active guardrail and quality-gate proof.

