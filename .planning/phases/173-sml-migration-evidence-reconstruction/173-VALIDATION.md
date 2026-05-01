---
phase: 173
slug: sml-migration-evidence-reconstruction
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-05-01
---

# Phase 173 — Validation Strategy

## Quick Feedback Lane

- `git log -p -n 2 -- cmake/sml_version.cmake`
- `rg -n '#\s*include\s*[<"](boost/sml|sml\.hpp|boost/sml\.hpp)' src include tests tools docs scripts .codex/get-shit-done cmake CMakeLists.txt || true`
- `rg -n '\bboost::sml\b|using\s+namespace\s+boost::sml|#\s*include\s*[<"]boost/sml' src include tests tools docs scripts .codex/get-shit-done cmake CMakeLists.txt || true`
- `rg -n '#\s*include\s*[<"]stateforward/sml|stateforward::sml' src include tests tools docs/rules tools/docsgen CMakeLists.txt cmake`

## Rule Compliance Evidence

- Guards/actions were not modified.
- Runtime behavior was not changed.
- Evidence is based on live source and git history, not planning artifacts alone.

## Notes

No unresolved escalations or manual-only blockers remain for this phase.

