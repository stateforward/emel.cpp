---
phase: 36-parity-and-regression-proof
plan: 01
subsystem: publication-plumbing
tags: [embedded-size, docsgen, snapshot, readme, publication]
requires:
  - phase: 35-maintained-runtime-execution-on-arm
    provides: matched comparator row and smoke proof
provides:
  - executable-size snapshot schema and docsgen pipeline
  - generated README publication path
  - explicit scope wording for matched Qwen3 E2E executable comparison
affects: [39-01 publication refresh]
tech-stack:
  added: []
  patterns: [snapshot-driven docs, generated readme, narrow publication wording]
key-files:
  created: []
  modified:
    [scripts/embedded_size.sh, tools/docsgen/docsgen.cpp, docs/templates/README.md.j2, README.md, snapshots/embedded_size/summary.txt]
key-decisions:
  - "Publication is driven from the stored snapshot through docsgen, not by hand-edited README text."
  - "The wording stays on matched Qwen3 E2E executable comparison and avoids whole-product parity claims."
requirements-completed: []
duration: 0min
completed: 2026-04-02
---

# Phase 36 Plan 01 Summary

**The executable-size publication plumbing is proven and remains scope-limited**

## Accomplishments

- Verified the maintained executable-size metadata surface in
  [scripts/embedded_size.sh](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/scripts/embedded_size.sh)
  emits the raw, stripped, section, workload, and smoke fields used by the milestone.
- Verified the generated README publication path through
  [docsgen.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/docsgen/docsgen.cpp),
  [README.md.j2](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/docs/templates/README.md.j2),
  and
  [README.md](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/README.md).
- Preserved narrow publication wording for the matched Qwen3 E2E executable comparison.

## Verification

- `./scripts/embedded_size.sh --json`
- `./build/docsgen/docsgen --root /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp --check`

## Deviations from Plan

- Snapshot freshness is deferred to Phase 39 because the publication pipeline exists, but the
  checked-in snapshot still needs to be refreshed from the latest local executable-size run.

---
*Phase: 36-parity-and-regression-proof*
*Completed: 2026-04-02*
