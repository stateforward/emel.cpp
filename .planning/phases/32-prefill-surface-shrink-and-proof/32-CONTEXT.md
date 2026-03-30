---
phase: 32
slug: prefill-surface-shrink-and-proof
created: 2026-03-29
status: ready
---

# Phase 32 Context

## Phase Boundary

Finish the milestone by shrinking the remaining top-level generator surface and rerunning the
maintained regression/proof lane, including a performance check.

## Implementation Decisions

### Surface Shrink
- Remove parent-generator prefill-only actions and guards that became dead after the child
  extraction.
- Keep decode behavior and the public generator wrapper unchanged.
- Refresh generated architecture docs so the parent/child boundary is visible in the maintained
  diagrams.

### Proof
- Rerun maintained generator and parity coverage plus the repo gate.
- Measure maintained generation benchmark impact against the pre-refactor baseline with the same
  low-iteration compare lane.
- Report performance impact honestly even if the result is “no meaningful change.”

## Existing Code Insights

- `src/emel/generator/actions.hpp` and `src/emel/generator/guards.hpp` still contain prefill-only
  helpers that the parent no longer references after Phase 31.
- `src/emel/generator/sm.hpp` is already materially smaller, but the generated docs still reflect
  the old layout until `scripts/generate_docs.sh` is rerun.
- The maintained generator contract tests already passed after the child extraction.

## Specific Ideas

- Remove unused parent prefill helpers first, then rerun build/tests to ensure the shrink really
  has no behavior drift.
- Use the maintained generation-only compare lane for the perf check so the reported delta matches
  the repo’s operator-facing surface.
