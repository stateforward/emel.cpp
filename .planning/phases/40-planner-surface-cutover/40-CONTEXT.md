---
phase: 40
slug: planner-surface-cutover
created: 2026-04-04
status: ready
---

# Phase 40 Context

## Phase Boundary

Phase 40 hard-cuts only the top-level planner surface under `src/emel/batch/planner/` so a
maintainer can find the planner entrypoint, wrapper, and planner-owned orchestration from one
canonical family path.

This phase is not the place to reorganize the child mode directories, rename planner-family events,
or redesign how per-dispatch data moves through the planner family. Those changes belong to phases
41 through 43.

## Implementation Decisions

### Planner Entry Surface
- Keep `src/emel/batch/planner/sm.hpp` as the canonical planner machine entrypoint.
- Preserve canonical `emel::batch::planner::sm` and additive PascalCase `emel::batch::planner::Planner`.
- Do not introduce additional top-level planner type aliases or split the entry surface across
  multiple non-canonical files in this phase.

### Phase Boundary Discipline
- Limit structural work in this phase to the top-level planner wrapper, its immediate top-level
  file bases, and direct consumers that reference that wrapper.
- Leave child mode directory cleanup and file-base canonicalization for phase 41.
- Leave planner-family event renames, typed handoff cleanup, and context/runtime ownership changes
  for phases 42 and 43.

### Integration Touchpoints
- Update only direct planner consumers that rely on the top-level planner type or its includes,
  specifically the generator path, `src/emel/machines.hpp`, planner-focused tests, and planner
  benches.
- Keep the current request/callback contract behaviorally stable in phase 40 unless a narrow
  top-level rename is required to preserve the canonical entrypoint.

### Surface Readability
- Planner-owned orchestration should remain understandable from planner-family files without
  introducing new helper surfaces outside `src/emel/batch/planner/`.
- Existing shared planner-mode helpers may stay in place temporarily if moving them would pull
  phase 40 into the child-mode cleanup reserved for phase 41.

### Verification Stance
- Phase 40 should leave focused planner tests able to prove the top-level surface still wires the
  maintained batching flow.
- Full behavior-preservation proof and milestone validation closeout remain phase 44 work; this
  phase only needs enough focused verification to show the surface cutover is correctly integrated.

## Specific Ideas

- Keep the planner request/callback API recognizable to current generator and bench callers during
  phase 40; broader contract renaming is intentionally deferred.
- Prefer top-level readability over opportunistic cleanup in unrelated planner-mode files.

## Canonical References

### Contract Sources
- `AGENTS.md` — hard-cut file-base, naming, event, context, and workflow contract for this
  milestone.
- `docs/rules/sml.rules.md` — authoritative SML actor-model and run-to-completion semantics.

### Milestone Scope
- `.planning/ROADMAP.md` — phase boundary, dependencies, and success criteria for v1.10.
- `.planning/REQUIREMENTS.md` — `PLAN-01`, `PLAN-02`, and `PLAN-03` traceability for this phase.
- `.planning/PROJECT.md` — current milestone scope and the proof-first project constraint.

### Existing Planner Surface
- `src/emel/batch/planner/sm.hpp` — current top-level planner wrapper and transition table.
- `src/emel/batch/planner/events.hpp` — current planner request/runtime/outcome surface that phase
  40 must avoid widening.
- `src/emel/generator/actions.hpp` — primary maintained caller that dispatches planner requests.
- `src/emel/machines.hpp` — existing public additive alias surface touched by planner naming.

## Existing Code Insights

### Reusable Assets
- `src/emel/batch/planner/` already has the canonical top-level file bases
  `actions/context/errors/events/guards/sm`, so the cutover can focus on naming and direct
  consumer cleanup instead of inventing a new structure.
- `tests/batch/planner/` already exercises the planner wrapper and transition flow, giving this
  phase a focused proof surface for top-level integration.

### Established Patterns
- The planner wrapper currently owns dispatch-local `request_ctx` and forwards work into child mode
  submachines through SML completion transitions.
- Generator and bench callers instantiate `emel::batch::planner::sm` directly, so any top-level
  surface adjustment must keep those maintained call sites coherent.

### Integration Points
- `src/emel/generator/context.hpp`
- `src/emel/generator/actions.hpp`
- `src/emel/machines.hpp`
- `tools/bench/batch/planner_bench.cpp`
- `tests/batch/planner/*.cpp`

## Deferred Ideas

- Canonicalize `simple`, `equal`, and `sequential` mode directory/file surfaces in phase 41.
- Replace planner/mode reach-through with explicit typed handoff and contract-aligned outcome
  events in phase 42.
- Remove remaining per-dispatch planner runtime/control data from non-canonical locations in phase
  43.
- Land milestone-level behavior-preservation proof and x86 validation closeout in phase 44.
