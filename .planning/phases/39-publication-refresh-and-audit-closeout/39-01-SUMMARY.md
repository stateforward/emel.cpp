---
phase: 39-publication-refresh-and-audit-closeout
plan: 01
subsystem: publication-closeout
tags: [embedded-size, publication, audit, qwen3, docs]
requires:
  - phase: 38-retroactive-traceability-and-proof-backfill
    provides: auditable v1.8 phase chain
provides:
  - refreshed executable-size snapshot and generated README evidence
  - completed publication proof for PUB-01 and PUB-02
  - updated v1.8 milestone audit on the narrowed comparator scope
affects: [v1.8 closeout readiness, milestone audit]
tech-stack:
  added: []
  patterns: [snapshot refresh, generated publication proof, milestone closeout]
key-files:
  created:
    [.planning/phases/39-publication-refresh-and-audit-closeout/39-VERIFICATION.md]
  modified:
    [snapshots/embedded_size/summary.txt, README.md, .planning/REQUIREMENTS.md, .planning/ROADMAP.md, .planning/STATE.md, .planning/v1.8-v1.8-MILESTONE-AUDIT.md]
key-decisions:
  - "The closeout publication stays on one maintained EMEL-versus-llama.cpp executable row pair and does not widen comparator scope."
  - "The stale v1.8 audit should be replaced with a fresh eight-requirement audit instead of patched incrementally."
requirements-completed: [PUB-01, PUB-02]
duration: 0min
completed: 2026-04-02
---

# Phase 39 Plan 01 Summary

**The published executable-size evidence now matches the corrected local flow**

## Accomplishments

- Refreshed the checked-in executable-size snapshot and generated
  [README.md](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/README.md) so the published
  values now match the corrected EMEL probe path: `4,073,016` raw bytes, `4,073,016` stripped
  bytes, and `1,323,877` section bytes versus the matched `llama.cpp` reference row at
  `3,334,264`, `2,795,112`, and `3,094,255`.
- Checked off all v1.8 requirements in
  [REQUIREMENTS.md](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/REQUIREMENTS.md)
  and updated traceability so the milestone no longer reports pending publication requirements.
- Replaced the stale pre-backfill
  [v1.8-v1.8-MILESTONE-AUDIT.md](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/v1.8-v1.8-MILESTONE-AUDIT.md)
  with a fresh audit on the narrowed EMEL-versus-`llama.cpp` scope.
- Marked Phase `39` complete in
  [ROADMAP.md](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/ROADMAP.md) and
  advanced
  [STATE.md](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/STATE.md) to
  milestone-audit-ready status.

## Verification

- `./scripts/embedded_size.sh --json`
- `./build/docsgen/docsgen --root /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp --check`
- `./scripts/quality_gates.sh`
- `node ~/.codex/get-shit-done/bin/gsd-tools.cjs roadmap analyze`

## Deviations from Plan

- The full quality gate passed only after carrying forward the already-applied paritychecker
  compatibility patch for upstream `llama.cpp` scale-field renames. That was verifier work, not a
  scope expansion of publication closeout.

---
*Phase: 39-publication-refresh-and-audit-closeout*
*Completed: 2026-04-02*
