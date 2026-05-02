---
phase: 168
slug: project-owned-source-namespace-migration
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-01
---

# Phase 168 — Validation Strategy

## Quick Feedback Lane

- `scripts/check_legacy_sml_surface.sh`
- `rg -n '#\s*include\s*[<"]stateforward/sml|stateforward::sml' src include tests tools docs/rules tools/docsgen CMakeLists.txt cmake`

## Rule Compliance Evidence

- Evidence-only backfill; no runtime behavior was changed in this phase directory.
- Phase 173 contains the reconstructed live proof.

