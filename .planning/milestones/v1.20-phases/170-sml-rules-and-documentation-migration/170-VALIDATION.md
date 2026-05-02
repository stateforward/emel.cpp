---
phase: 170
slug: sml-rules-and-documentation-migration
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-01
---

# Phase 170 — Validation Strategy

## Quick Feedback Lane

- `rg -n 'docs/sml\.rules\.md' . || true`
- `scripts/generate_docs.sh --check`
- `scripts/lint_snapshot.sh`

## Rule Compliance Evidence

- Documentation-only backfill; Phase 175 contains the live repair evidence.

