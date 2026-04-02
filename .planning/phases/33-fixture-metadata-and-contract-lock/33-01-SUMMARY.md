---
phase: 33-fixture-metadata-and-contract-lock
plan: 01
subsystem: workload-boundary
tags: [planning, workload, publication, comparator]
requires:
  - phase: 32-prefill-surface-shrink-and-proof
    provides: prior shipped milestone baseline and active v1.8 planning branch
provides:
  - locked maintained Qwen3 executable-size workload boundary
  - explicit executable-only publication claim for v1.8
  - narrowed comparator scope to EMEL and one matched llama.cpp row
affects: [34-03 emel probe proof, 35-01 comparator smoke proof, 39-01 publication refresh]
tech-stack:
  added: []
  patterns: [planning truth source, scope guardrails, narrow comparator boundary]
key-files:
  created: []
  modified:
    [.planning/PROJECT.md, .planning/REQUIREMENTS.md, .planning/ROADMAP.md, .planning/STATE.md]
key-decisions:
  - "v1.8 is fixed to tests/models/Qwen3-0.6B-Q8_0.gguf on the structured hello -> first-token path."
  - "Final linked executables are the maintained truth surface; library artifact size is not."
  - "LiteRT is removed from active v1.8 scope; the published comparison is EMEL versus one matched llama.cpp reference row."
requirements-completed: []
duration: 0min
completed: 2026-04-02
---

# Phase 33 Plan 01 Summary

**The v1.8 workload and claim boundary are now locked in planning**

## Accomplishments

- Updated
  [PROJECT.md](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/PROJECT.md),
  [REQUIREMENTS.md](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/REQUIREMENTS.md),
  [ROADMAP.md](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/ROADMAP.md),
  and
  [STATE.md](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/STATE.md)
  so v1.8 consistently names the maintained fixture, structured `hello` input, and
  `max_tokens=1` generation slice.
- Made final linked executables the only maintained v1.8 truth surface.
- Removed LiteRT from the active milestone scope and narrowed the published comparator set to EMEL
  and one matched `llama.cpp` reference row.

## Verification

- `rg -n 'Qwen3-0.6B-Q8_0.gguf|hello|max_tokens=1|final linked executables|llama.cpp' .planning/PROJECT.md .planning/REQUIREMENTS.md .planning/ROADMAP.md .planning/STATE.md`

## Deviations from Plan

- None in scope. This phase stayed declarative and only tightened milestone truth surfaces.

---
*Phase: 33-fixture-metadata-and-contract-lock*
*Completed: 2026-04-02*
